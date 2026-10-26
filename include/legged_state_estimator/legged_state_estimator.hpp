#ifndef LEGGED_STATE_ESTIMATOR_LEGGED_STATE_ESTIMATOR_HPP_
#define LEGGED_STATE_ESTIMATOR_LEGGED_STATE_ESTIMATOR_HPP_

#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"

#include "legged_state_estimator/inekf/inekf.hpp"
#include "legged_state_estimator/inekf/inekf_state.hpp"
#include "legged_state_estimator/inekf/noise_params.hpp"
#include "legged_state_estimator/inekf/observations.hpp"
#include "legged_state_estimator/robot_model.hpp"
#include "legged_state_estimator/contact_estimator.hpp"
#include "legged_state_estimator/low_pass_filter.hpp"
#include "legged_state_estimator/legged_state_estimator_settings.hpp"


namespace legged_state_estimator {

///
/// @class LeggedStateEstimator
/// @brief State estimator for legged robots.
///
class LeggedStateEstimator {
public:
  using Vector3d = Eigen::Matrix<double, 3, 1>;
  using Vector4d = Eigen::Matrix<double, 4, 1>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix3d = Eigen::Matrix<double, 3, 3>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

  ///
  /// @brief Constructor.
  /// @param[in] settings State estimator settings.
  ///
  LeggedStateEstimator(const LeggedStateEstimatorSettings& settings);

  ///
  /// @brief Default constructor.
  ///
  LeggedStateEstimator();

  ///
  /// @brief Default destructor.
  ///
  ~LeggedStateEstimator();

  LeggedStateEstimator(const LeggedStateEstimator&) = default;
  LeggedStateEstimator& operator=(const LeggedStateEstimator&) = default;
  LeggedStateEstimator(LeggedStateEstimator&&) noexcept = default;
  LeggedStateEstimator& operator=(LeggedStateEstimator&&) noexcept = default;

  ///
  /// @brief Initializes the state estimator.
  /// @param[in] base_pos Base position. 
  /// @param[in] base_quat Base orientation expressed by quaternion (x, y, z, w). 
  /// @param[in] base_lin_vel_world Base linear velocity expressed in the world
  /// coordinate. Default is Eigen::Vector3d::Zero().
  /// @param[in] imu_gyro_bias Initial guess of the IMU gyro bias. Default is 
  /// Eigen::Vector3d::Zero().
  /// @param[in] imu_lin_accel_bias Initial guess of the IMU linear acceleration 
  /// bias. Default is Eigen::Vector3d::Zero().
  ///
  void init(const Eigen::Vector3d& base_pos, const Eigen::Vector4d& base_quat,
            const Eigen::Vector3d& base_lin_vel_world=Eigen::Vector3d::Zero(),
            const Eigen::Vector3d& imu_gyro_bias=Eigen::Vector3d::Zero(),
            const Eigen::Vector3d& imu_lin_accel_bias=Eigen::Vector3d::Zero());

  ///
  /// @brief Initializes the state estimator.
  /// @param[in] base_pos Base position. 
  /// @param[in] base_quat Base orientation expressed by quaternion (x, y, z, w). 
  /// @param[in] qJ Raw measurement of the joint positions. 
  /// @param[in] ground_height Ground height. 
  /// @param[in] base_lin_vel_world Base linear velocity expressed in the world
  /// coordinate. Default is Eigen::Vector3d::Zero().
  /// @param[in] imu_gyro_bias Initial guess of the IMU gyro bias. Default is 
  /// Eigen::Vector3d::Zero().
  /// @param[in] imu_lin_accel_bias Initial guess of the IMU linear acceleration 
  /// bias. Default is Eigen::Vector3d::Zero().
  ///
  void init(const Eigen::Vector3d& base_pos, const Eigen::Vector4d& base_quat,
            const Eigen::VectorXd& qJ, 
            const std::vector<double>& ground_height={0., 0., 0., 0.},
            const Eigen::Vector3d& base_lin_vel_world=Eigen::Vector3d::Zero(),
            const Eigen::Vector3d& imu_gyro_bias=Eigen::Vector3d::Zero(),
            const Eigen::Vector3d& imu_lin_accel_bias=Eigen::Vector3d::Zero());

  ///
  /// @brief Updates the state estimation.
  /// @param[in] imu_gyro_raw Raw measurement of the base angular velocity 
  /// expressed in the body local coordinate from IMU gyro sensor.
  /// @param[in] imu_lin_accel_raw Raw measurement of the base linear 
  /// acceleration expressed in the body local coordinate from IMU accelerometer. 
  /// @param[in] qJ Raw measurement of the joint positions. 
  /// @param[in] dqJ Raw measurement of the joint velocities. 
  /// @param[in] tauJ Raw measurement of the joint torques. 
  ///
  void update(const Eigen::Vector3d& imu_gyro_raw, 
              const Eigen::Vector3d& imu_lin_accel_raw, 
              const Eigen::VectorXd& qJ, const Eigen::VectorXd& dqJ, 
              const Eigen::VectorXd& tauJ);

  ///
  /// @brief Updates the state estimation using externally supplied contact states.
  /// @param[in] contact_state Contact ID and active state for every configured contact.
  ///
  void update(const Eigen::Vector3d& imu_gyro_raw,
              const Eigen::Vector3d& imu_lin_accel_raw,
              const Eigen::VectorXd& qJ, const Eigen::VectorXd& dqJ,
              const Eigen::VectorXd& tauJ,
              const std::vector<std::pair<int, bool>>& contact_state);

  ///
  /// @return const reference to the base position estimate.
  ///
  const Eigen::Vector3d& getBasePositionEstimate() const;

  ///
  /// @return const reference to the base orientation estimate expressed by a 
  /// rotation matrix.
  ///
  const Eigen::Matrix3d& getBaseRotationEstimate() const;

  ///
  /// @return const reference to the base orientation estimate expressed by 
  /// quaternion.
  ///
  const Eigen::Vector4d& getBaseQuaternionEstimate() const;

  ///
  /// @return const reference to the base linear velocity estimate expressed in 
  /// the world frame.
  ///
  const Eigen::Vector3d& getBaseLinearVelocityEstimateWorld() const;

  ///
  /// @return const reference to the base linear velocity estimate expressed in 
  /// the body local coordinate.
  ///
  const Eigen::Vector3d getBaseLinearVelocityEstimateLocal() const;

  ///
  /// @return const reference to the base angular velocity estimate expressed in 
  /// the world frame.
  ///
  const Eigen::Vector3d& getBaseAngularVelocityEstimateWorld() const;

  ///
  /// @return const reference to the base angular velocity estimate expressed in 
  /// the local frame.
  ///
  const Eigen::Vector3d& getBaseAngularVelocityEstimateLocal() const;

  ///
  /// @return const reference to the IMU gyro bias estimate. 
  ///
  const Eigen::Vector3d& getIMUGyroBiasEstimate() const;

  ///
  /// @return const reference to the IMU linear acceleration bias estimate. 
  ///
  const Eigen::Vector3d& getIMULinearAccelerationBiasEstimate() const;

  ///
  /// @return const reference to the joint velocity estimates. 
  ///
  const Eigen::VectorXd& getJointVelocityEstimate() const;

  ///
  /// @return const reference to the joint acceleration estimates. 
  ///
  const Eigen::VectorXd& getJointAccelerationEstimate() const;

  ///
  /// @return const reference to the joint torque estimates. 
  ///
  const Eigen::VectorXd& getJointTorqueEstimate() const;

  ///
  /// @return const reference to the conatct estimator. 
  ///
  const ContactEstimator& getContactEstimator() const;

  ///
  /// @return const reference to the state estimator settings.
  ///
  const LeggedStateEstimatorSettings& getSettings() const;

  ///
  /// @return const reference to the underlying InEKF, for diagnostics that need
  /// the raw filter internals (covariance blocks, augmented contact bookkeeping).
  ///
  const InEKF& getInEKF() const;

  ///
  /// @return diagnostics of the most recent InEKF measurement update: whether it
  /// fired, the innovation and its predicted covariance, the NIS, and which
  /// contacts were corrected / augmented / marginalized. Valid until the next
  /// update().
  ///
  const CorrectionInfo& getLastCorrection() const;

  ///
  /// @brief Per-block standard deviations of the InEKF covariance, i.e. the
  /// square roots of the diagonals of the attitude, velocity, position,
  /// gyroscope-bias and accelerometer-bias blocks of P. These are the cheapest
  /// signal that separates "the filter is confident and right" from "the filter
  /// is confident and wrong".
  ///
  Eigen::Vector3d getAttitudeStdDev() const;
  Eigen::Vector3d getVelocityStdDev() const;
  Eigen::Vector3d getPositionStdDev() const;
  Eigen::Vector3d getGyroscopeBiasStdDev() const;
  Eigen::Vector3d getAccelerometerBiasStdDev() const;

  ///
  /// @return number of contact landmarks currently augmented into the filter
  /// state. Differs from the number of scheduled contacts during the update in
  /// which a contact is added or removed.
  ///
  int getNumAugmentedContacts() const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  void updateImpl(const Eigen::Vector3d& imu_gyro_raw,
                  const Eigen::Vector3d& imu_lin_accel_raw,
                  const Eigen::VectorXd& qJ, const Eigen::VectorXd& dqJ,
                  const Eigen::VectorXd& tauJ,
                  const std::vector<std::pair<int, bool>>* external_contact_state);

  LeggedStateEstimatorSettings settings_;
  InEKF inekf_;
  vectorKinematics leg_kinematics_;
  RobotModel robot_model_;
  ContactEstimator contact_estimator_;
  std::vector<Eigen::Vector3d> surface_normals_;
  std::vector<double> contact_stance_time_;  // seconds each contact has been in stance
  LowPassFilter<double, 3> lpf_gyro_accel_world_, lpf_lin_accel_world_;
  LowPassFilter<double, Eigen::Dynamic> lpf_dqJ_, lpf_ddqJ_, lpf_tauJ_;
  Vector3d imu_gyro_raw_world_, imu_gyro_raw_world_prev_, imu_gyro_accel_world_, 
           imu_gyro_accel_local_, imu_lin_accel_raw_world_, imu_lin_accel_local_,
           base_pos_estimate_, base_lin_vel_world_estimate_, base_lin_vel_local_estimate_,
           base_ang_vel_world_estimate_, base_ang_vel_local_estimate_,
           imu_gyro_bias_estimate_, imu_lin_acc_bias_estimate_;
  Matrix3d base_rot_estimate_;
  Vector6d imu_raw_;
  Vector4d base_quat_estimate_;

};

} // namespace legged_state_estimator

#endif // LEGGED_STATE_ESTIMATOR_LEGGED_STATE_ESTIMATOR_HPP_
