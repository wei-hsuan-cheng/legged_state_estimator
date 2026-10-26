#ifndef LEGGED_STATE_ESTIMATOR_CONTACT_ESTIMATOR_HPP_
#define LEGGED_STATE_ESTIMATOR_CONTACT_ESTIMATOR_HPP_

#include <vector>
#include <utility>
#include <iostream>

#include "Eigen/Core"
#include "Eigen/Cholesky"
#include "Eigen/StdVector"

#include "legged_state_estimator/robot_model.hpp"


namespace legged_state_estimator {

///
/// @class ContactEstimatorSettings
/// @brief Settings of the contact estimator.
///
struct ContactEstimatorSettings {
  ///
  /// @brief Offset of the logistic regression for the contact state (on/off), one per contact.
  /// A contact group (rigid body) uses the values of its first contact.
  ///
  std::vector<double> beta0;

  ///
  /// @brief Slope of the logistic regression on the group's normal force [1/N], one per contact.
  ///
  std::vector<double> beta1;

  ///
  /// @brief A parameter in computing the contact force covariances.
  ///
  double contact_force_covariance_alpha;

  ///
  /// @brief A group enters stance when its contact probability reaches this value.
  ///
  double contact_probability_threshold;

  ///
  /// @brief A group in stance leaves it when its probability falls below this value
  /// (hysteresis). A negative value means contact_probability_threshold, i.e. no hysteresis.
  ///
  double contact_release_probability_threshold = -1.0;

  ///
  /// @brief Rigid-body (group) index of each contact. Contacts on one body, e.g. the heel and
  /// toe of a flat foot, share one stance decision made on their summed normal force. Empty
  /// means every contact is its own group (point feet).
  ///
  std::vector<int> contact_groups;

  ///
  /// @brief A contact of a group in stance counts as in contact only while it carries at least
  /// this fraction of its group's normal force. 0 keeps every contact of a group in stance
  /// together. A positive value lets a flat foot roll: the toe is not a fixed point before it
  /// is loaded at a heel strike, nor the heel after it lifts at toe-off.
  ///
  double contact_point_force_ratio = 0.0;

  ///
  /// @brief Damping of the least-squares contact-force solve [m^2]; keeps the split between
  /// contacts of one body well defined when the joint torques cannot separate them.
  ///
  double force_regularization = 1.0e-6;
};


///
/// @class ContactEstimator
/// @brief Contact estimator from joint torques, for any number of contacts and joints.
///
/// The joint rows of the floating-base dynamics give tau - tau_ID = -Jc^T f, with Jc the
/// stacked 3D contact Jacobians over the joints and tau_ID the inverse dynamics without
/// contact. All contact forces are solved together by damped least squares, so no joint
/// layout is assumed: a contact's force comes from whichever joints actually move it.
///
class ContactEstimator {
public:
  ///
  /// @brief Constructor.
  /// @param[in] robot_model Robot model.
  /// @param[in] settings Contact estimator settings.
  ///
  ContactEstimator(const RobotModel& robot_model, 
                   const ContactEstimatorSettings& settings);

  ///
  /// @brief Default constructor.
  ///
  ContactEstimator();

  ///
  /// @brief Default destructor.
  ///
  ~ContactEstimator() = default;

  ContactEstimator(const ContactEstimator&) = default;
  ContactEstimator& operator=(const ContactEstimator&) = default;
  ContactEstimator(ContactEstimator&&) noexcept = default;
  ContactEstimator& operator=(ContactEstimator&&) noexcept = default;

  ///
  /// @brief Resets the estimation.
  ///
  void reset();

  ///
  /// @brief Updates the estimation.
  /// @param[in] robot_model Robot model.
  /// @param[in] tauJ Current joint torque.
  ///
  void update(const RobotModel& robot_model, const Eigen::VectorXd& tauJ);

  ///
  /// @brief Set the parameters.
  /// @param[in] settings Settings.
  ///
  void setParameters(const ContactEstimatorSettings& settings);

  ///
  /// @brief Get the contact state, i.e., on/off of the contacts.
  ///
  const std::vector<std::pair<int, bool>>& getContactState() const;

  ///
  /// @brief Get the contact probabilities.
  /// @return const reference to the contact probabilities.
  ///
  const std::vector<double>& getContactProbability() const;

  ///
  /// @brief Get the contact force covariances.
  /// @return const reference to the contact force covariances.
  ///
  const std::vector<double>& getContactForceCovariance() const;

  ///
  /// @brief Get the contact force estimates.
  /// @return const reference to the contact force estimates.
  ///
  const std::vector<Eigen::Vector3d>& getContactForceEstimate() const;

  ///
  /// @brief Get the normal contact force estimates.
  /// @return const reference to the normal contact force estimates.
  ///
  const std::vector<double>& getNormalContactForceEstimate() const;

  ///
  /// @brief Get the contact normal surface.
  /// @return const reference to the contact normal surface.
  ///
  const std::vector<Eigen::Vector3d>& getContactSurfaceNormal() const;

  ///
  /// @brief Number of contact groups (rigid bodies).
  ///
  int numGroups() const { return num_groups_; }

  ///
  /// @brief Summed normal contact force of each group [N].
  ///
  const std::vector<double>& getGroupNormalForceEstimate() const { return group_normal_force_; }

  ///
  /// @brief Stance decision of each group.
  ///
  const std::vector<bool>& getGroupContactState() const { return group_state_; }

  ///
  /// @brief Sets the contact normal surface.
  /// @param[in] Contact Contact normal vectors.
  ///
  void setContactSurfaceNormal(const std::vector<Eigen::Vector3d>& contact_surface_normal);

  void disp(std::ostream& os) const;

  friend std::ostream& operator<<(std::ostream& os, const ContactEstimator& d);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  ContactEstimatorSettings settings_;
  std::vector<Eigen::Vector3d> contact_force_estimate_, 
                               contact_force_estimate_prev_, 
                               contact_surface_normal_;
  std::vector<double> normal_contact_force_estimate_, 
                      normal_contact_force_estimate_prev_, 
                      contact_probability_, contact_force_covariance_;
  std::vector<std::pair<int, bool>> contact_state_;
  int num_contacts_;
  int num_groups_ = 0;
  std::vector<int> group_of_contact_, first_contact_of_group_;
  std::vector<double> group_normal_force_, group_normal_force_prev_;
  std::vector<bool> group_state_;
  Eigen::MatrixXd jacobian_, normal_matrix_;
  Eigen::VectorXd residual_, projected_residual_, forces_;
  Eigen::LDLT<Eigen::MatrixXd> solver_;
};

} // namespace legged_state_estimator

#endif // LEGGED_STATE_ESTIMATOR_CONTACT_ESTIMATOR_HPP_ 