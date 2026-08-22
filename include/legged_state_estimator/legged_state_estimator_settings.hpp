#ifndef LEGGED_STATE_ESTIMATOR_LEGGED_STATE_ESTIMATOR_SETTINGS_HPP_
#define LEGGED_STATE_ESTIMATOR_LEGGED_STATE_ESTIMATOR_SETTINGS_HPP_

#include <string>
#include <vector>

#include "legged_state_estimator/inekf/noise_params.hpp"
#include "legged_state_estimator/contact_estimator.hpp"


namespace legged_state_estimator {

///
/// @class LeggedStateEstimatorSettings
/// @brief Settings of the legged state estimator.
///
struct LeggedStateEstimatorSettings {
public:
  /// 
  /// @brief Path to the URDF file.
  ///
  std::string urdf_path;

  /// 
  /// @brief Name of the IMU frame specified in the URDF file.
  ///
  std::string imu_frame;

  /// 
  /// @brief Nemes of the contact frames specified in the URDF file.
  ///
  std::vector<std::string> contact_frames;

  /// 
  /// @brief Contact estimator settings. 
  ///
  ContactEstimatorSettings contact_estimator_settings;

  /// 
  /// @brief Noise parameters (covariances) of InEKF. 
  ///
  NoiseParams inekf_noise_params;

  /// 
  /// @brief Use dynamics in contact estimation. If false, equilibrium is 
  /// used for contact estimation. Default is false.
  ///
  bool dynamic_contact_estimation = false;

  /// 
  /// @brief Noise (covariance) on contact position. (Possibly is not used in 
  /// InEKF. Contact covariance in noise_params are more important).
  ///
  double contact_position_noise;

  /// 
  /// @brief Noise (covariance) on contact rotation. Only used with surface 
  /// contacts.
  ///
  double contact_rotation_noise;

  ///
  /// @brief Time step of estimation.
  ///
  double sampling_time;

  ///
  /// @brief Initial standard deviations of the InEKF state, i.e. how much the
  /// pose passed to init() is trusted. These matter more than they look: the
  /// underlying InEKFState defaults to P = I(15), which claims 1 rad of attitude
  /// uncertainty, 1 m/s of velocity and 1 m of position even when the filter was
  /// just seeded from a known pose. The first contact correction, whose noise is
  /// millimetres, then yanks the state hard - a transient usually mistaken for
  /// slow bias convergence.
  ///
  /// Seeded from ground truth (simulation): make these tight.
  /// Seeded on hardware: tight roll/pitch (gravity is observable when standing
  /// still), loose xy/yaw (unobservable - they are a gauge freedom).
  ///
  /// @{
  double initial_attitude_noise = 1.0e-2;            //!< [rad]
  double initial_velocity_noise = 1.0e-2;            //!< [m/s]
  double initial_position_noise = 1.0e-2;            //!< [m]
  double initial_gyroscope_bias_noise = 1.0e-2;      //!< [rad/s]
  double initial_accelerometer_bias_noise = 1.0e-1;  //!< [m/s^2]
  /// @}

  ///
  /// @brief Estimate the IMU biases. When false the bias block is decoupled from
  /// the rest of the covariance on every propagation, freezing the bias at the
  /// value passed to init().
  ///
  /// Note the bias is observable only through its cross-covariance with the
  /// attitude/velocity/position block - the kinematic measurement Jacobian has no
  /// bias columns - so the initial bias covariance above has to be non-zero for
  /// this to do anything.
  ///
  bool estimate_imu_bias = true;

  /// 
  /// @brief Cutoff frequency of LPF for gyro sensor. 
  ///
  double lpf_gyro_cutoff_frequency;

  /// 
  /// @brief Cutoff frequency of LPF for gyro acceleration that is computed by
  /// finite difference approximation. 
  ///
  double lpf_gyro_accel_cutoff_frequency;

  /// 
  /// @brief Cutoff frequency of LPF for linear acceleration measurement from IMU. 
  ///
  double lpf_lin_accel_cutoff_frequency;

  /// 
  /// @brief Cutoff frequency of LPF for joint velocities. 
  ///
  double lpf_dqJ_cutoff_frequency;

  /// 
  /// @brief Cutoff frequency of LPF for joint accelerations. 
  ///
  double lpf_ddqJ_cutoff_frequency;

  /// 
  /// @brief Cutoff frequency of LPF for joint torques. 
  ///
  double lpf_tauJ_cutoff_frequency;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// 
  /// @brief Creates settings for Unitree A1. 
  /// @param[in] urdf_path Path to the URDF file.
  /// @param[in] sampling_time Sampling time.
  ///
  static LeggedStateEstimatorSettings UnitreeA1(const std::string& urdf_path, 
                                                const double sampling_time);

};

} // namespace legged_state_estimator

#endif // LEGGED_STATE_ESTIMATOR_LEGGED_STATE_ESTIMATOR_SETTINGS_HPP_