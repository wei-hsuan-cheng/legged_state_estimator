#include "legged_state_estimator/legged_state_estimator.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>


namespace legged_state_estimator {

LeggedStateEstimator::LeggedStateEstimator(const LeggedStateEstimatorSettings& settings)
  : settings_(settings),
    inekf_(settings.inekf_noise_params),
    leg_kinematics_(),
    robot_model_(settings.urdf_path, settings.imu_frame, settings.contact_frames),
    contact_estimator_(robot_model_, settings.contact_estimator_settings),
    lpf_gyro_accel_world_(settings.sampling_time, settings.lpf_gyro_accel_cutoff_frequency),
    lpf_lin_accel_world_(settings.sampling_time, settings.lpf_lin_accel_cutoff_frequency),
    lpf_dqJ_(settings.sampling_time, settings.lpf_dqJ_cutoff_frequency, robot_model_.nJ()),
    lpf_ddqJ_(settings.sampling_time, settings.lpf_ddqJ_cutoff_frequency, robot_model_.nJ()),
    lpf_tauJ_(settings.sampling_time, settings.lpf_tauJ_cutoff_frequency, robot_model_.nJ()),
    imu_gyro_raw_world_(Vector3d::Zero()), 
    imu_gyro_raw_world_prev_(Vector3d::Zero()), 
    imu_gyro_accel_world_(Vector3d::Zero()), 
    imu_gyro_accel_local_(Vector3d::Zero()), 
    imu_lin_accel_raw_world_(Vector3d::Zero()), 
    imu_lin_accel_local_(Vector3d::Zero()),
    base_pos_estimate_(Vector3d::Zero()),
    base_lin_vel_world_estimate_(Vector3d::Zero()), 
    base_lin_vel_local_estimate_(Vector3d::Zero()),
    base_ang_vel_world_estimate_(Vector3d::Zero()), 
    base_ang_vel_local_estimate_(Vector3d::Zero()),
    imu_gyro_bias_estimate_(Vector3d::Zero()), 
    imu_lin_acc_bias_estimate_(Vector3d::Zero()),
    base_rot_estimate_(Matrix3d::Identity()),
    imu_raw_(Vector6d::Zero()),
    base_quat_estimate_(Eigen::Quaterniond::Identity().coeffs()) {
  if (settings.sampling_time <= 0.0) {
    throw std::invalid_argument(
        "[LeggedStateEstimator] invalid argment: sampling_time must be positive");
  }
  const double contact_position_cov = settings.contact_position_noise * settings.contact_position_noise;
  const double contact_rotation_cov = settings.contact_rotation_noise * settings.contact_rotation_noise;
  Matrix6d cov_leg = Matrix6d::Zero();
  cov_leg.topLeftCorner<3, 3>() = contact_position_cov * Eigen::Matrix3d::Identity();
  cov_leg.bottomRightCorner<3, 3>() = contact_rotation_cov * Eigen::Matrix3d::Identity();
  for (int i=0; i<settings.contact_frames.size(); ++i) {
    leg_kinematics_.emplace_back(i, Eigen::Matrix4d::Identity(), cov_leg);
  }
  imu_raw_.setZero();
}


LeggedStateEstimator::LeggedStateEstimator() 
  : settings_(),
    inekf_(),
    leg_kinematics_(),
    robot_model_(),
    contact_estimator_(),
    lpf_gyro_accel_world_(),
    lpf_lin_accel_world_(),
    lpf_dqJ_(),
    lpf_ddqJ_(),
    lpf_tauJ_(),
    imu_gyro_raw_world_(Vector3d::Zero()), 
    imu_gyro_raw_world_prev_(Vector3d::Zero()), 
    imu_gyro_accel_world_(Vector3d::Zero()), 
    imu_gyro_accel_local_(Vector3d::Zero()), 
    imu_lin_accel_raw_world_(Vector3d::Zero()), 
    imu_lin_accel_local_(Vector3d::Zero()),
    base_pos_estimate_(Vector3d::Zero()),
    base_lin_vel_world_estimate_(Vector3d::Zero()), 
    base_lin_vel_local_estimate_(Vector3d::Zero()),
    base_ang_vel_world_estimate_(Vector3d::Zero()), 
    base_ang_vel_local_estimate_(Vector3d::Zero()),
    imu_gyro_bias_estimate_(Vector3d::Zero()), 
    imu_lin_acc_bias_estimate_(Vector3d::Zero()),
    base_rot_estimate_(Matrix3d::Identity()),
    imu_raw_(Vector6d::Zero()),
    base_quat_estimate_(Eigen::Quaterniond::Identity().coeffs()) {
}


LeggedStateEstimator::~LeggedStateEstimator() {}


void LeggedStateEstimator::init(const Eigen::Vector3d& base_pos,
                                const Eigen::Vector4d& base_quat,
                                const Eigen::Vector3d& base_lin_vel_world,
                                const Eigen::Vector3d& imu_gyro_bias,
                                const Eigen::Vector3d& imu_lin_accel_bias) {
  InEKFState initial_state;
  initial_state.setPosition(base_pos);
  initial_state.setRotation(Eigen::Quaterniond(base_quat).toRotationMatrix());
  initial_state.setVelocity(base_lin_vel_world);
  initial_state.setGyroscopeBias(imu_gyro_bias);
  initial_state.setAccelerometerBias(imu_lin_accel_bias);

  // The initial covariance says how much the seed above is trusted. Leaving it
  // at the InEKFState default of identity claims sigma = 1 rad on attitude,
  // 1 m/s on velocity and 1 m on position for a state that was just seeded from
  // a known pose; the first contact correction, whose noise is millimetres, then
  // yanks the state hard. Settings carry explicit std-devs instead.
  const auto square = [](const double sigma) { return sigma * sigma; };
  initial_state.setRotationCovariance(
      square(settings_.initial_attitude_noise) * Eigen::Matrix3d::Identity());
  initial_state.setVelocityCovariance(
      square(settings_.initial_velocity_noise) * Eigen::Matrix3d::Identity());
  initial_state.setPositionCovariance(
      square(settings_.initial_position_noise) * Eigen::Matrix3d::Identity());
  initial_state.setGyroscopeBiasCovariance(
      square(settings_.initial_gyroscope_bias_noise) * Eigen::Matrix3d::Identity());
  initial_state.setAccelerometerBiasCovariance(
      square(settings_.initial_accelerometer_bias_noise) * Eigen::Matrix3d::Identity());

  // resetState(), not setState(): the augmented contact/landmark maps index into
  // the state matrix X, so a stale entry left from a previous activation would
  // index past the end of the freshly assigned 5x5 X on the next correction.
  // That made any deactivate -> activate cycle undefined behaviour.
  inekf_.resetState(initial_state);
  inekf_.setEstimateBias(settings_.estimate_imu_bias);

  imu_gyro_raw_world_.setZero();
  imu_gyro_raw_world_prev_.setZero();
  imu_gyro_accel_world_.setZero();
  imu_gyro_accel_local_.setZero();
  imu_lin_accel_raw_world_.setZero();
  imu_lin_accel_local_.setZero();

  lpf_gyro_accel_world_.reset();
  lpf_lin_accel_world_.reset();
  lpf_dqJ_.reset();
  lpf_ddqJ_.reset();
  lpf_tauJ_.reset();
  contact_estimator_.reset();
  contact_stance_time_.assign(contact_stance_time_.size(), 0.0);
}


void LeggedStateEstimator::init(const Eigen::Vector3d& base_pos,
                                const Eigen::Vector4d& base_quat,
                                const Eigen::VectorXd& qJ, 
                                const std::vector<double>& ground_height,
                                const Eigen::Vector3d& base_lin_vel_world,
                                const Eigen::Vector3d& imu_gyro_bias,
                                const Eigen::Vector3d& imu_lin_accel_bias) {
  if (qJ.size() != robot_model_.nJ()) {
    throw std::invalid_argument(
        "[LeggedStateEstimator] invalid argment: qJ.size() must be " + std::to_string(robot_model_.nJ()));
  }
  if (ground_height.size() != robot_model_.numContacts()) {
    throw std::invalid_argument(
        "[LeggedStateEstimator] invalid argment: ground_height.size() must be " + std::to_string(robot_model_.numContacts()));
  }
  robot_model_.updateKinematics(Eigen::Vector3d::Zero(), base_quat, qJ);
  double base_height = 0;
  for (int i=0; i<robot_model_.numContacts(); ++i) {
    base_height += (robot_model_.getBasePosition().coeff(2)
                      - robot_model_.getContactPosition(i).coeff(2)
                      + ground_height[i]);
  }
  base_height /= robot_model_.numContacts();
  const Eigen::Vector3d base_pos_correct 
      = {base_pos.coeff(0), base_pos.coeff(1), base_height};
  init(base_pos_correct, base_quat, base_lin_vel_world, 
       imu_gyro_bias, imu_lin_accel_bias);
}


void LeggedStateEstimator::update(const Eigen::Vector3d& imu_gyro_raw, 
                                  const Eigen::Vector3d& imu_lin_accel_raw, 
                                  const Eigen::VectorXd& qJ, 
                                  const Eigen::VectorXd& dqJ, 
                                  const Eigen::VectorXd& tauJ) {
  updateImpl(imu_gyro_raw, imu_lin_accel_raw, qJ, dqJ, tauJ, nullptr);
}


void LeggedStateEstimator::update(
    const Eigen::Vector3d& imu_gyro_raw,
    const Eigen::Vector3d& imu_lin_accel_raw,
    const Eigen::VectorXd& qJ,
    const Eigen::VectorXd& dqJ,
    const Eigen::VectorXd& tauJ,
    const std::vector<std::pair<int, bool>>& contact_state) {
  updateImpl(imu_gyro_raw, imu_lin_accel_raw, qJ, dqJ, tauJ, &contact_state);
}


void LeggedStateEstimator::updateImpl(
    const Eigen::Vector3d& imu_gyro_raw,
    const Eigen::Vector3d& imu_lin_accel_raw,
    const Eigen::VectorXd& qJ,
    const Eigen::VectorXd& dqJ,
    const Eigen::VectorXd& tauJ,
    const std::vector<std::pair<int, bool>>* external_contact_state) {
  if (qJ.size() != robot_model_.nJ()) {
    throw std::invalid_argument(
        "[LeggedStateEstimator] invalid argment: qJ.size() must be " + std::to_string(robot_model_.nJ()));
  }
  if (dqJ.size() != robot_model_.nJ()) {
    throw std::invalid_argument(
        "[LeggedStateEstimator] invalid argment: dqJ.size() must be " + std::to_string(robot_model_.nJ()));
  }
  if (tauJ.size() != robot_model_.nJ()) {
    throw std::invalid_argument(
        "[LeggedStateEstimator] invalid argment: tauJ.size() must be " + std::to_string(robot_model_.nJ()));
  }
  if (external_contact_state != nullptr) {
    if (external_contact_state->size() != static_cast<size_t>(robot_model_.numContacts())) {
      throw std::invalid_argument(
          "[LeggedStateEstimator] external contact_state must have one entry per configured contact");
    }
    std::vector<bool> contact_id_seen(static_cast<size_t>(robot_model_.numContacts()), false);
    for (const auto& contact : *external_contact_state) {
      const int contact_id = contact.first;
      if (contact_id < 0 || contact_id >= robot_model_.numContacts() ||
          contact_id_seen[static_cast<size_t>(contact_id)]) {
        throw std::invalid_argument(
            "[LeggedStateEstimator] external contact_state IDs must be unique and in range");
      }
      contact_id_seen[static_cast<size_t>(contact_id)] = true;
    }
  }
  // Process IMU measurements in InEKF
  imu_raw_.template head<3>() = imu_gyro_raw;
  imu_raw_.template tail<3>() = imu_lin_accel_raw;
  inekf_.Propagate(imu_raw_, settings_.sampling_time);
  // Process IMU measurements in LPFs (linear acceleration)
  imu_lin_accel_raw_world_.noalias() = getBaseRotationEstimate() * (imu_lin_accel_raw - getIMULinearAccelerationBiasEstimate());
  lpf_lin_accel_world_.update(imu_lin_accel_raw_world_);
  imu_lin_accel_local_.noalias() = getBaseRotationEstimate().transpose() * lpf_lin_accel_world_.getEstimate();
  // Process IMU measurements in LPFs (angular acceleration via a finite difference)
  if (settings_.dynamic_contact_estimation) {
    imu_gyro_raw_world_.noalias() = getBaseRotationEstimate() * (imu_gyro_raw - getIMUGyroBiasEstimate());
    imu_gyro_accel_world_.noalias() = (imu_gyro_raw_world_ - imu_gyro_raw_world_prev_) / settings_.sampling_time;
    lpf_gyro_accel_world_.update(imu_gyro_accel_world_);
    imu_gyro_accel_local_.noalias() = getBaseRotationEstimate().transpose() * lpf_gyro_accel_world_.getEstimate();
    imu_gyro_raw_world_prev_ = imu_gyro_raw_world_;
  }
  // Process joint measurements in LPFs 
  if (settings_.dynamic_contact_estimation) {
    lpf_ddqJ_.update((dqJ-lpf_dqJ_.getEstimate())/settings_.sampling_time);
  }
  lpf_dqJ_.update(dqJ);
  lpf_tauJ_.update(tauJ);
  // Update contact info
  robot_model_.updateLegKinematics(qJ);
  // Torque-based contact estimate. It drives the contacts when none are supplied and runs as
  // a diagnostic otherwise, so a detector can be compared against supplied contacts live.
  // Inverse dynamics at the estimated base attitude, so gravity loads the joints in the
  // right direction even when the trunk leans. The accelerometer measures the specific
  // force f = R^T (a - g); Pinocchio adds -g itself, so the base acceleration it needs is
  // f + R^T g (zero when the base is at rest), not f.
  const Eigen::Vector3d gravity_local =
      getBaseRotationEstimate().transpose() * Eigen::Vector3d(0.0, 0.0, -9.81);
  if (settings_.dynamic_contact_estimation) {
    robot_model_.updateDynamics(Eigen::Vector3d::Zero(), getBaseQuaternionEstimate(),
                                getBaseLinearVelocityEstimateLocal(),
                                imu_gyro_raw - getIMUGyroBiasEstimate(),
                                imu_lin_accel_local_ + gravity_local, imu_gyro_accel_local_,
                                qJ, lpf_dqJ_.getEstimate(), lpf_ddqJ_.getEstimate());
  }
  else {
    robot_model_.updateDynamics(Eigen::Vector3d::Zero(), getBaseQuaternionEstimate(),
                                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                                qJ, lpf_dqJ_.getEstimate(), Eigen::VectorXd::Zero(robot_model_.nJ()));
  }
  // The contact Jacobians are taken with the base at the identity (updateLegKinematics), so
  // forces are in the base frame and the ground normal there is R^T e_z.
  surface_normals_.assign(static_cast<size_t>(robot_model_.numContacts()),
                          getBaseRotationEstimate().transpose().col(2));
  contact_estimator_.setContactSurfaceNormal(surface_normals_);
  contact_estimator_.update(robot_model_, lpf_tauJ_.getEstimate());
  if (external_contact_state != nullptr) {
    inekf_.setContacts(*external_contact_state);
  }
  else {
    inekf_.setContacts(contact_estimator_.getContactState());
  }
  const double minimum_contact_covariance =
      settings_.contact_position_noise * settings_.contact_position_noise;
  const double settle_covariance =
      settings_.contact_settle_position_noise * settings_.contact_settle_position_noise;
  const auto& used_contacts = external_contact_state != nullptr ?
      *external_contact_state : contact_estimator_.getContactState();
  contact_stance_time_.resize(static_cast<size_t>(robot_model_.numContacts()), 0.0);
  for (const auto& [id, in_stance] : used_contacts) {
    double& stance_time = contact_stance_time_[static_cast<size_t>(id)];
    stance_time = in_stance ? stance_time + settings_.sampling_time : 0.0;
  }
  for (int i=0; i<robot_model_.numContacts(); ++i) {
    leg_kinematics_[i].setContactPosition(
        robot_model_.getContactPosition(i)-robot_model_.getBasePosition());
    double contact_force_cov = external_contact_state != nullptr ?
        minimum_contact_covariance :
        std::max(minimum_contact_covariance,
                 contact_estimator_.getContactForceCovariance()[i]);
    if (contact_stance_time_[i] > 0.0 && contact_stance_time_[i] <= settings_.contact_settle_time) {
      contact_force_cov = std::max(contact_force_cov, settle_covariance);
    }
    leg_kinematics_[i].setContactPositionCovariance(
        contact_force_cov*Eigen::Matrix3d::Identity());
  }
  // Process kinematics measurements in InEKF
  inekf_.CorrectKinematics(leg_kinematics_);
  // Restore estimates
  base_pos_estimate_ = inekf_.getState().getPosition();
  base_rot_estimate_ = inekf_.getState().getRotation();
  base_quat_estimate_ = Eigen::Quaterniond(inekf_.getState().getRotation()).coeffs();
  base_lin_vel_world_estimate_ = inekf_.getState().getVelocity();
  base_lin_vel_local_estimate_ = inekf_.getState().getRotation().transpose() * base_lin_vel_world_estimate_;
  base_ang_vel_local_estimate_ = imu_gyro_raw - inekf_.getState().getGyroscopeBias();
  // This used to be assigned imu_gyro_raw_world_, which is only ever written
  // inside the dynamic_contact_estimation branch above - so with that option off
  // (the common case) the world angular velocity was reported as exactly zero
  // forever. Derive it from the local estimate and the corrected rotation, which
  // is also more accurate than the pre-correction value that variable holds.
  base_ang_vel_world_estimate_.noalias() = base_rot_estimate_ * base_ang_vel_local_estimate_;
  imu_gyro_bias_estimate_ = inekf_.getState().getGyroscopeBias();
  imu_lin_acc_bias_estimate_ = inekf_.getState().getAccelerometerBias();
}


const Eigen::Vector3d& LeggedStateEstimator::getBasePositionEstimate() const {
  return base_pos_estimate_;
}


const Eigen::Matrix3d& LeggedStateEstimator::getBaseRotationEstimate() const {
  return base_rot_estimate_;
}


const Eigen::Vector4d& LeggedStateEstimator::getBaseQuaternionEstimate() const {
  return base_quat_estimate_;
}


const Eigen::Vector3d& LeggedStateEstimator::getBaseLinearVelocityEstimateWorld() const {
  return base_lin_vel_world_estimate_;
}


const Eigen::Vector3d LeggedStateEstimator::getBaseLinearVelocityEstimateLocal() const {
  return base_lin_vel_local_estimate_;
}


const Eigen::Vector3d& LeggedStateEstimator::getBaseAngularVelocityEstimateWorld() const {
  return base_ang_vel_world_estimate_;
}


const Eigen::Vector3d& LeggedStateEstimator::getBaseAngularVelocityEstimateLocal() const {
  return base_ang_vel_local_estimate_;
}


const Eigen::Vector3d& LeggedStateEstimator::getIMUGyroBiasEstimate() const {
  return imu_gyro_bias_estimate_;
}


const Eigen::Vector3d& LeggedStateEstimator::getIMULinearAccelerationBiasEstimate() const {
  return imu_lin_acc_bias_estimate_;
}


const Eigen::VectorXd& LeggedStateEstimator::getJointVelocityEstimate() const {
  return lpf_dqJ_.getEstimate();
}


const Eigen::VectorXd& LeggedStateEstimator::getJointAccelerationEstimate() const {
  return lpf_ddqJ_.getEstimate();
}


const Eigen::VectorXd& LeggedStateEstimator::getJointTorqueEstimate() const {
  return lpf_tauJ_.getEstimate();
}


const ContactEstimator& LeggedStateEstimator::getContactEstimator() const {
  return contact_estimator_;
}


const LeggedStateEstimatorSettings& LeggedStateEstimator::getSettings() const {
  return settings_;
}


const InEKF& LeggedStateEstimator::getInEKF() const {
  return inekf_;
}


const CorrectionInfo& LeggedStateEstimator::getLastCorrection() const {
  return inekf_.getLastCorrection();
}


namespace {
// The covariance blocks are symmetric positive semi-definite in exact
// arithmetic, but a diagonal entry can go marginally negative through round-off;
// clamp before the square root so a diagnostic never returns NaN.
Eigen::Vector3d blockStdDev(const Eigen::Block<const Eigen::MatrixXd, 3, 3>& block) {
  return block.diagonal().cwiseMax(0.0).cwiseSqrt();
}
} // namespace


Eigen::Vector3d LeggedStateEstimator::getAttitudeStdDev() const {
  return blockStdDev(inekf_.getState().getRotationCovariance());
}


Eigen::Vector3d LeggedStateEstimator::getVelocityStdDev() const {
  return blockStdDev(inekf_.getState().getVelocityCovariance());
}


Eigen::Vector3d LeggedStateEstimator::getPositionStdDev() const {
  return blockStdDev(inekf_.getState().getPositionCovariance());
}


Eigen::Vector3d LeggedStateEstimator::getGyroscopeBiasStdDev() const {
  return blockStdDev(inekf_.getState().getGyroscopeBiasCovariance());
}


Eigen::Vector3d LeggedStateEstimator::getAccelerometerBiasStdDev() const {
  return blockStdDev(inekf_.getState().getAccelerometerBiasCovariance());
}


int LeggedStateEstimator::getNumAugmentedContacts() const {
  return static_cast<int>(inekf_.getEstimatedContactPositions().size());
}

} // namespace legged_state_estimator
