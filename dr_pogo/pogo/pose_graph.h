#pragma once

#include <vector>
#include <map>
#include <array>
#include <stdint.h>
#include <ceres/ceres.h>
#include <utils.h>


struct PoseGraphOpts
{
    double loss_scale_loop_pos_coarse = 5.0;  // in [m]
    double loss_scale_loop_pos_fine = 1.0;  // in [m]
    double loss_scale_loop_rot = 0.05;  // in [rad]

    double odom_pos_std = 0.025;    // in [m]
    double odom_rot_std = 0.00002;  // in [rad]
    double loop_pos_std = 0.05;     // in [m]
    double loop_rot_std = 0.0002;   // in [rad]
    double bias_std = 1e-12;       // in [rad]
    
    bool estimate_bias = false;

};


class PoseGraph {
    public:
        PoseGraph(const PoseGraphOpts& opts);

        void addOdometryEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose, double bias_prior);

        void addLoopClosurePosEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose);
        void addLoopClosureRotEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose);
        void addLoopClosureEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose);

        void optimize();


        void printPoses() const;

        void printLastPose() const;

        size_t size() const { return node_times_.size(); }

        void writeToFile(const std::string& filename) const;

        std::pair<std::vector<int64_t>, std::vector<std::array<double, 3>>> getPoses() const
        {
            std::vector<int64_t> times;
            std::vector<std::array<double, 3>> poses;
            for(size_t i = 0; i < node_times_.size(); ++i)
            {
                times.push_back(node_times_[i]);
                poses.push_back(*(node_poses_[i]));
            }
            return {times, poses};
        }

    private:
        PoseGraphOpts opts_;

        // Storing state variables and timestamps
        std::map<int64_t, size_t> node_indices_;
        std::vector<int64_t> node_times_;
        std::vector<std::shared_ptr<std::array<double, 3>>> node_poses_;
        std::vector<std::shared_ptr<double>> node_biases_;
        std::vector<double> bias_priors_;


        // Loss functions
        ceres::LossFunction* loss_function_loop_pos_ = nullptr;
        ceres::LossFunction* loss_function_loop_rot_ = nullptr;
        ceres::LossFunction* loss_function_odom_pos_ = nullptr;
        ceres::LossFunction* loss_function_odom_rot_ = nullptr;

        ceres::Problem problem_;

};




class RelativeRotCostFunction: public ceres::SizedCostFunction<1, 3, 3>
{
    public:
        RelativeRotCostFunction(const std::array<double, 3>& relative_pose, double weight = 1.0);

        virtual ~RelativeRotCostFunction() {}

        virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;
    private:
        Eigen::Matrix3d inv_meas_;
        double weight_;
};



class RelativePosCostFunction : public ceres::SizedCostFunction<2, 3, 3>
{
    public:
        RelativePosCostFunction(const std::array<double, 3>& relative_pose, double weight = 1.0);

        virtual ~RelativePosCostFunction() {}

        virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;
    private:
        Eigen::Vector2d meas_;
        double weight_;
};



class RelativePoseWithBiasCostFunction : public ceres::SizedCostFunction<3, 3, 3, 1>
{
    public:
        RelativePoseWithBiasCostFunction(const std::array<double, 3>& relative_pose, double pos_weight, double rot_weight, double bias_prior, double delta_t);

        virtual ~RelativePoseWithBiasCostFunction() {}

        virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

    private:
        std::array<double, 3> relative_meas_;
        double pos_weight_;
        double rot_weight_;
        double bias_prior_;
        double delta_t_;
};


class BrownianMotionCostFunction : public ceres::SizedCostFunction<1, 1, 1>
{
    public:
        BrownianMotionCostFunction(double weight, double bias_prior_1, double bias_prior_2);

        virtual ~BrownianMotionCostFunction() {}

        virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

    private:
        double weight_;
        double bias_prior_1_;
        double bias_prior_2_;
};


class DynamicCauchyLoss : public ceres::LossFunction {
    public:
        explicit DynamicCauchyLoss(double a) : a_(a) {}

        // Setter function to update scale
        void setScale(double new_scale) {
            a_ = new_scale;
        }

        double getScale() const {
            return a_;
        }

        void Evaluate(double s, double* rho) const override {
            const double sum = 1.0 + s / (a_ * a_);
            rho[0] = a_ * a_ * log(sum);           
            rho[1] = 1.0 / sum;                   
            rho[2] = -rho[1] / sum / (a_ * a_);   
        }

    private:
        double a_;  // Mutable scale parameter
};