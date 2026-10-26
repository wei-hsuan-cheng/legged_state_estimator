#include "legged_state_estimator/contact_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>


namespace legged_state_estimator {

ContactEstimator::ContactEstimator(const RobotModel& robot_model,
                                   const ContactEstimatorSettings& settings)
  : settings_(settings),
    contact_force_estimate_(robot_model.numContacts(), Eigen::Vector3d::Zero()),
    contact_force_estimate_prev_(robot_model.numContacts(), Eigen::Vector3d::Zero()),
    contact_surface_normal_(robot_model.numContacts(), Eigen::Vector3d::UnitZ()),
    normal_contact_force_estimate_(robot_model.numContacts(), 0),
    normal_contact_force_estimate_prev_(robot_model.numContacts(), 0),
    contact_probability_(robot_model.numContacts(), 0),
    contact_force_covariance_(robot_model.numContacts(), 0),
    contact_state_(),
    num_contacts_(robot_model.numContacts()) {
  const auto n = static_cast<size_t>(num_contacts_);
  if (settings.beta0.size() != n || settings.beta1.size() != n) {
    throw std::invalid_argument(
        "[ContactEstimator] invalid argment: settings.beta0 and beta1 need one entry per contact ("
        + std::to_string(n) + ")");
  }
  if (settings.contact_probability_threshold <= 0 || settings.contact_probability_threshold >= 1.0) {
    throw std::invalid_argument(
        "[ContactEstimator] invalid argment: settings.contact_probability_threshold must be in (0, 1)");
  }
  if (settings.contact_release_probability_threshold >= 1.0) {
    throw std::invalid_argument(
        "[ContactEstimator] invalid argment: settings.contact_release_probability_threshold must be below 1");
  }
  if (settings.force_regularization < 0) {
    throw std::invalid_argument(
        "[ContactEstimator] invalid argment: settings.force_regularization must be non-negative");
  }
  // Groups: renumber densely so any labels work, e.g. foot indices [0, 0, 1, 1].
  if (!settings.contact_groups.empty() && settings.contact_groups.size() != n) {
    throw std::invalid_argument(
        "[ContactEstimator] invalid argment: settings.contact_groups needs one entry per contact or none");
  }
  group_of_contact_.assign(n, 0);
  std::vector<int> labels;
  for (size_t i = 0; i < n; ++i) {
    const int label = settings.contact_groups.empty() ? static_cast<int>(i) : settings.contact_groups[i];
    const auto it = std::find(labels.begin(), labels.end(), label);
    group_of_contact_[i] = static_cast<int>(it - labels.begin());
    if (it == labels.end()) {
      labels.push_back(label);
      first_contact_of_group_.push_back(static_cast<int>(i));
    }
  }
  num_groups_ = static_cast<int>(labels.size());
  group_normal_force_.assign(labels.size(), 0.0);
  group_normal_force_prev_.assign(labels.size(), 0.0);
  group_state_.assign(labels.size(), false);

  const int nJ = robot_model.nJ();
  jacobian_.setZero(3 * num_contacts_, nJ);
  normal_matrix_.setZero(3 * num_contacts_, 3 * num_contacts_);
  residual_.setZero(nJ);
  projected_residual_.setZero(3 * num_contacts_);
  forces_.setZero(3 * num_contacts_);
  solver_ = Eigen::LDLT<Eigen::MatrixXd>(3 * num_contacts_);
  contact_state_.clear();
  for (int i=0; i<num_contacts_; ++i) {
    contact_state_.push_back(std::pair<int, bool>(i, false));
  }
}


ContactEstimator::ContactEstimator() 
  : settings_(),
    contact_force_estimate_(),
    contact_force_estimate_prev_(),
    contact_surface_normal_(),
    normal_contact_force_estimate_(),
    normal_contact_force_estimate_prev_(),
    contact_probability_(),
    contact_force_covariance_(),
    contact_state_(),
    num_contacts_(0) {
}


void ContactEstimator::reset() {
  std::fill(group_state_.begin(), group_state_.end(), false);
  std::fill(group_normal_force_prev_.begin(), group_normal_force_prev_.end(), 0.0);
  for (auto& e : contact_state_) {
    e.second = false;
  }
}


void ContactEstimator::update(const RobotModel& robot_model, 
                              const Eigen::VectorXd& tauJ) {
  // Contact forces from the joint rows of the dynamics: tau - tau_ID = -Jc^T f, solved for
  // all contacts at once as f = -(Jc Jc^T + lambda I)^-1 Jc (tau - tau_ID).
  for (int i=0; i<num_contacts_; ++i) {
    jacobian_.middleRows(3*i, 3) = robot_model.getJointContactJacobian(i);
  }
  residual_ = tauJ - robot_model.getJointInverseDynamics();
  normal_matrix_.noalias() = jacobian_ * jacobian_.transpose();
  normal_matrix_.diagonal().array() += settings_.force_regularization;
  projected_residual_.noalias() = jacobian_ * residual_;
  solver_.compute(normal_matrix_);
  forces_ = -solver_.solve(projected_residual_);

  for (int i=0; i<num_contacts_; ++i) {
    contact_force_estimate_[i] = forces_.segment<3>(3*i);
    normal_contact_force_estimate_[i] = contact_force_estimate_[i].dot(contact_surface_normal_[i]);
  }
  std::fill(group_normal_force_.begin(), group_normal_force_.end(), 0.0);
  for (int i=0; i<num_contacts_; ++i) {
    group_normal_force_[group_of_contact_[i]] += normal_contact_force_estimate_[i];
  }
  // One decision per rigid body, on its summed normal force, with hysteresis.
  const double release = settings_.contact_release_probability_threshold < 0 ?
      settings_.contact_probability_threshold : settings_.contact_release_probability_threshold;
  for (int g=0; g<num_groups_; ++g) {
    const int first = first_contact_of_group_[g];
    double probability = 1.0 / (1.0 + std::exp(- settings_.beta1[first] * group_normal_force_[g]
                                                - settings_.beta0[first]));
    if (!std::isfinite(probability)) {
      probability = 0.0;
    }
    group_state_[g] = probability >= (group_state_[g] ? release : settings_.contact_probability_threshold);
    const double df = group_normal_force_[g] - group_normal_force_prev_[g];
    group_normal_force_prev_[g] = group_normal_force_[g];
    const double point_force = settings_.contact_point_force_ratio * std::max(group_normal_force_[g], 0.0);
    for (int i=0; i<num_contacts_; ++i) {
      if (group_of_contact_[i] == g) {
        contact_probability_[i] = probability;
        contact_force_covariance_[i] = settings_.contact_force_covariance_alpha * df * df;
        contact_state_[i].second = group_state_[g] &&
            (settings_.contact_point_force_ratio <= 0.0 || normal_contact_force_estimate_[i] >= point_force);
      }
    }
  }
  for (int i=0; i<num_contacts_; ++i) {
    contact_force_estimate_prev_[i] = contact_force_estimate_[i];
    normal_contact_force_estimate_prev_[i] = normal_contact_force_estimate_[i];
  }
}


void ContactEstimator::setParameters(const ContactEstimatorSettings& settings) {
  settings_ = settings;
}


const std::vector<std::pair<int, bool>>& ContactEstimator::getContactState() const {
  return contact_state_;
}


const std::vector<double>& ContactEstimator::getContactProbability() const {
  return contact_probability_;
}


const std::vector<double>& ContactEstimator::getContactForceCovariance() const {
  return contact_force_covariance_;
}


// double ContactEstimator::getContactForceCovariance() const {
//   int num_active_contacts = 0;
//   for (const auto e : getContactState()) {
//     if (e.second) {
//       ++num_active_contacts;
//     }
//   }
//   double contact_force_cov = 0;
//   for (int i=0; i<num_contacts_; ++i) {
//     if (getContactState()[i].second) {
//       contact_force_cov += contact_force_covariance_[i];
//     }
//   }
//   if (num_active_contacts > 0) {
//     contact_force_cov *= (1.0/num_active_contacts);
//   }
//   else {
//     contact_force_cov = 0;
//   }
//   return contact_force_cov;
// }


const std::vector<Eigen::Vector3d>& ContactEstimator::getContactForceEstimate() const {
  return contact_force_estimate_;
}


const std::vector<double>& ContactEstimator::getNormalContactForceEstimate() const {
  return normal_contact_force_estimate_;
}


const std::vector<Eigen::Vector3d>& ContactEstimator::getContactSurfaceNormal() const {
  return contact_surface_normal_;
}


void ContactEstimator::setContactSurfaceNormal(
    const std::vector<Eigen::Vector3d>& contact_surface_normal) {
  assert(contact_surface_normal_.size() == contact_surface_normal.size());
  contact_surface_normal_ = contact_surface_normal;
}


void ContactEstimator::disp(std::ostream& os) const {
  os << "Contact estimation:" << std::endl;
  os << "  contact state: [";
  for (int i=0; i<num_contacts_-1; ++i) {
    os << std::boolalpha << getContactState()[i].second << ", ";
  }
  os << std::boolalpha << getContactState()[num_contacts_-1].second << "]" << std::endl;
  /////////////////////////////////////////
  os << "  contact probability: [";
  for (int i=0; i<num_contacts_-1; ++i) {
    os << getContactProbability()[i] << ", ";
  }
  os << getContactProbability()[num_contacts_-1] << "]" << std::endl;
  /////////////////////////////////////////
  os << "  contact force covariance: [";
  for (int i=0; i<num_contacts_-1; ++i) {
    os << getContactForceCovariance()[i] << ", ";
  }
  os << getContactForceCovariance()[num_contacts_-1] << "]" << std::endl;
  /////////////////////////////////////////
  os << "  contact force estimate: [";
  for (int i=0; i<num_contacts_-1; ++i) {
    os << "[" << getContactForceEstimate()[i].transpose() << "], ";
  }
  os << "[" << getContactForceEstimate()[num_contacts_-1].transpose() << "]]" << std::endl;
  /////////////////////////////////////////
  os << "  normal contact force estimate: [";
  for (int i=0; i<num_contacts_-1; ++i) {
    os << getNormalContactForceEstimate()[i] << ", ";
  }
  os << getNormalContactForceEstimate()[num_contacts_-1] << "]" << std::endl;
  /////////////////////////////////////////
  os << "  assumed contact surface normal: [";
  for (int i=0; i<num_contacts_-1; ++i) {
    os << "[" << getContactSurfaceNormal()[i].transpose() << "], ";
  }
  os << "[" << getContactSurfaceNormal()[num_contacts_-1].transpose() << "]]" << std::flush;
}


std::ostream& operator<<(std::ostream& os, const ContactEstimator& e) {
  e.disp(os);
  return os;
}


} // namespace legged_state_estimator
