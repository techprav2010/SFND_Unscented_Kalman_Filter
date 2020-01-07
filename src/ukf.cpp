#include "ukf.h"
#include "Eigen/Dense"

#include <iostream>

using namespace std;
using std::cout;
using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;

const double _2_PI = 2.*M_PI;

/**
 * Constructor
 */
UKF::UKF() {

    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    /**
     * DO NOT MODIFY measurement noise values below.
     * These are provided by the sensor manufacturer.
     */

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    /**
     * End DO NOT MODIFY section for measurement noise values
     */
    is_initialized_ = false;
    // prev time zero
    time_us_ = 0;

    n_x_ = 5; // State dimension: [pos1 pos2 vel_abs yaw_angle yaw_rate]
    x_ = VectorXd::Zero(n_x_); //state vector - [pos1 pos2 vel_abs yaw_angle yaw_rate]
    P_ << 0.5,    0, 0, 0, 0,
            0, 0.5, 0, 0, 0,
            0,    0, 0.5, 0, 0,
            0,    0, 0, 1, 0,
            0,    0, 0, 0, 1;
    // Augmented state dimension
    n_aug_ = n_x_ + 2;
    // Sigma point spreading parameter
    lambda_ = 3 - n_aug_;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 2.8; //standard deviation longitudinal acceleration
    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 2.6;  //standard deviation yaw

    //sigma points weights
    weights_ = VectorXd::Zero(2 * n_aug_ + 1);
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    weights_.tail(2 * n_aug_ ).fill(0.5/(lambda_ + n_aug_));

    //sigma points
    Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1); //predicted sigma points
    Xsig_aug_ = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1); // agumented sigma points

    // noise measurement
    radar_R_ = MatrixXd(3, 3);
    radar_R_ << std_radr_*std_radr_,   0,                        0,
            0,                    std_radphi_*std_radphi_,  0,
            0,                    0,    std_radrd_*std_radrd_;
    lidar_R_ = MatrixXd(2, 2);
    lidar_R_ <<  std_laspx_*std_laspx_, 0,
            0, std_laspy_*std_laspy_;
    // NIS
    radar_NIS_ = 0.0;
    laser_NIS_= 0.0;
}

/**
 * Destructor
 */
UKF::~UKF() {}

/**
 * ProcessMeasurement
 * @param meas_package The latest measurement data of either radar or laser
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

    bool isRadarReading = meas_package.sensor_type_ == MeasurementPackage::RADAR;
    bool isLidarReading = meas_package.sensor_type_ == MeasurementPackage::LASER;

    if ( !isRadarReading  && !isLidarReading) {
        cout << "invalid sensor " << endl;
        return;
    }
    if ( isRadarReading  && !use_radar_) return;
    if ( isLidarReading && !use_laser_) return;


    if (!is_initialized_){
        if (isRadarReading){
            double rho = meas_package.raw_measurements_[0];      // range: radial distance from origin
            double phi = meas_package.raw_measurements_[1];      // bearing: angle between rho and x axis
            //double rho_dot = meas_package.raw_measurements_[2];  // radial velocity: change of rho
            double px = cos(phi) * rho;
            double py = sin(phi) * rho;
            double rhodot = meas_package.raw_measurements_[2];
            double v = sqrt(rhodot*sin(phi)*rhodot*sin(phi)
                     + rhodot*cos(phi)*rhodot*cos(phi));
            x_ << px, py, v, 0, 0;
        } else if  (isLidarReading){
            x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
        }

        time_us_ = meas_package.timestamp_; //Initialize the timestamp
        is_initialized_ = true;
        cout << "Done initializing, skip predict on first reading " << endl;
    }

    //Prediction
    double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0; //seconds
    time_us_ = meas_package.timestamp_;
    Prediction(delta_t); //Prediction
    if (isLidarReading){
        UpdateLidar(meas_package); // UpdateLidar
    } else if (isRadarReading){
        UpdateRadar(meas_package); // UpdateRadar
    }
}

/**
 * Prediction Predicts sigma points, the state, and the state covariance
 * matrix
 * @param delta_t Time between k and k+1 in s
 */
void UKF::Prediction(double delta_t) {
    /**
     * TODO: Complete this function! Estimate the object's location.
     * Modify the state vector, x_. Predict sigma points, the state,
     * and the state covariance matrix.
     */
    VectorXd x_aug = VectorXd(n_aug_);                  //augmented_mean
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);          //state augmented_covariance
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);   // sigma_point matrix

    //init augmented_mean
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    //init augmented_covariance
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5,5) = P_;
    P_aug(5,5) = std_a_*std_a_;
    P_aug(6,6) = std_yawdd_*std_yawdd_;
    MatrixXd L = P_aug.llt().matrixL();    // square root matrix

    //init sigma_point
    Xsig_aug.col(0) = x_aug;
    for(int i = 0; i < n_aug_; i++){
        Xsig_aug.col(i+1) = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
        Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
    }

    for(int i = 0; i<2*n_aug_+1; i++){
        // read into variables for readability
        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4,i);
        double nu_a = Xsig_aug(5,i);
        double nu_yawd = Xsig_aug(6,i);

        double px_p, py_p, v_p, yaw_p, yawd_p;
        if(fabs(yawd) > 0.001) { //division by zero, too small
            px_p = p_x + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
            py_p = p_y + v/yawd * (-cos(yaw + yawd*delta_t) + cos(yaw));
        } else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }
        v_p = v;
        yaw_p = yaw + yawd * delta_t;
        yawd_p = yawd;

        // noise
        px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p = v_p + nu_a * delta_t;
        yaw_p = yaw_p + 0.5 * nu_yawd * delta_t * delta_t;
        yawd_p = yawd_p + nu_yawd * delta_t;

        //update sigma points
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }

    x_.fill(0.0); // state mean
    for(int i = 0; i < 2*n_aug_+1; i++)
          x_ = x_ + weights_(i)*Xsig_pred_.col(i); // update state mean

    // x state covariance
    P_.fill(0.0);
    for(int i = 0; i < 2*n_aug_+1; i++)
    {
        VectorXd x_diff = Xsig_pred_.col(i) - x_; //residual
        while(x_diff(3) > M_PI)  x_diff(3) -= 2.*M_PI; //.. angle normalization
        while(x_diff(3) < -M_PI)  x_diff(3) += 2.*M_PI; //.. angle normalization
        P_ = P_ + weights_(i)*x_diff*x_diff.transpose(); //update  x state covariance
    }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement
 * @param meas_package The measurement at k+1
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
     * TODO: Complete this function! Use lidar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the lidar NIS, if desired.
     */

    int n_z_=2; //measurement dimension 2 : p_x and p_y
    MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);  //sigma points
    MatrixXd Tc = MatrixXd(n_x_, n_z_); //calculate cross correlation matrix
    MatrixXd S = MatrixXd(n_z_, n_z_); //measurement covariance matrix S
    VectorXd z_pred = VectorXd(n_z_); //mean predicted measurement

    //sigma points
    Zsig.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        Zsig(0, i) = Xsig_pred_(0, i);
        Zsig(1, i) = Xsig_pred_(1, i);
    }

    //mean predicted measurement
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;  //residual
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }
    S = S + lidar_R_; //add noise covariance matrix

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;  //residual
        VectorXd x_diff = Xsig_pred_.col(i) - x_;  // state difference
        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //update state mean and covariance
    VectorXd z = meas_package.raw_measurements_; //measurements
    MatrixXd K = Tc * S.inverse();  //kalman gain
    VectorXd z_diff = z - z_pred;   //residual
    x_ = x_ + K * z_diff; //update state mean
    P_ = P_ - K*S*K.transpose(); //update covariance matrix
    laser_NIS_= z_diff.transpose() * S.inverse() * z_diff; //calculate NIS

}

/**
 * Updates the state and the state covariance matrix using a radar measurement
 * @param meas_package The measurement at k+1
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
     * TODO: Complete this function! Use radar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the radar NIS, if desired.
     */
    int n_z_=3; //measurement dimension 3 : r, phi, and r_dot
    MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);  //sigma points
    MatrixXd Tc = MatrixXd(n_x_, n_z_); //calculate cross correlation matrix
    MatrixXd S = MatrixXd(n_z_, n_z_); //measurement covariance matrix S
    VectorXd z_pred = VectorXd(n_z_); //mean predicted measurement

    //sigma points
    Zsig.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v   = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);
        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;
        // model
        Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);                        //r
        Zsig(1, i) = atan2(p_y, p_x);                                 //phi
        Zsig(2, i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }

    //mean predicted measurement
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;  //residual
        while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;        //angle normalization
        while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;        //angle normalization
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }
    S = S + radar_R_; //add noise covariance matrix

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred; //residual
        while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI; //angle normalization
        while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;//angle normalization
        VectorXd x_diff = Xsig_pred_.col(i) - x_;  // state difference
        while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI; //angle normalization
        while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;//angle normalization
        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //update state mean and covariance
    VectorXd z = meas_package.raw_measurements_; //measurements
    MatrixXd K = Tc * S.inverse();  //kalman gain
    VectorXd z_diff = z - z_pred;   //residual
    while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI; //angle normalization
    while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI; //angle normalization
    x_ = x_ + K * z_diff; //update state mean
    P_ = P_ - K*S*K.transpose(); //update covariance matrix
    laser_NIS_= z_diff.transpose() * S.inverse() * z_diff; //calculate NIS

}

