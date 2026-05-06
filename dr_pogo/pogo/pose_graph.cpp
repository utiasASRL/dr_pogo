#include "pose_graph.h"
#include <iostream>
#include <cmath>
#include <stdexcept>




PoseGraph::PoseGraph(const PoseGraphOpts& opts)
    : opts_(opts)
{
    std::cout << "PoseGraph initialized with "
              << "\n\tloss_scale_loop_pos_coarse: " << opts.loss_scale_loop_pos_coarse << " m, "
              << "\n\tloss_scale_loop_pos_fine: " << opts.loss_scale_loop_pos_fine << " m, "
              << "\n\tloss_scale_loop_rot: " << opts.loss_scale_loop_rot << " rad, "
              << "\n\tstd_odom_pos: " << opts.odom_pos_std << " m, "
              << "\n\tstd_odom_rot: " << opts.odom_rot_std << " rad, "
              << "\n\tstd_loop_pos: " << opts.loop_pos_std << " m, "
              << "\n\tstd_loop_rot: " << opts.loop_rot_std << " rad" << std::endl;

    // Initialize loss functions
    loss_function_loop_pos_ = new DynamicCauchyLoss(opts.loss_scale_loop_pos_coarse / opts.loop_pos_std);
    loss_function_loop_rot_ = new ceres::CauchyLoss(opts.loss_scale_loop_rot / opts.loop_rot_std);
    //loss_function_odom_pos_ = new ceres::CauchyLoss(100.0);
    //loss_function_odom_rot_ = new ceres::CauchyLoss(100.0);
            
}



void PoseGraph::addOdometryEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose, double bias_prior)
{
    if(node_indices_.size() == 0)
    {
        node_times_.push_back(t0);
        node_poses_.push_back(std::make_shared<std::array<double, 3>>(std::array<double, 3>{0.0, 0.0, 0.0}));
        node_indices_[t0] = 0;

        problem_.AddParameterBlock(node_poses_[0]->data(), 3);
        problem_.SetParameterBlockConstant(node_poses_[0]->data());

    }

    if(t0 != node_times_.back())
    {
        throw std::runtime_error("Odometry edge t0 does not match the last node time.");
    }

    if(t1 <= t0)
    {
        throw std::runtime_error("Odometry edge t1 must be greater than t0.");
    }

    // Add the bias
    node_biases_.push_back(std::make_shared<double>(bias_prior));
    bias_priors_.push_back(bias_prior);
    problem_.AddParameterBlock(&(*(node_biases_.back())), 1);
    if(!opts_.estimate_bias)
    {
        problem_.SetParameterBlockConstant(&(*(node_biases_.back())));
    }


    // Add the new node
    node_indices_[t1] = node_poses_.size();
    node_times_.push_back(t1);
    std::array<double, 3> new_pose = combinePoses(*(node_poses_.back()), relative_pose);
    node_poses_.push_back(std::make_shared<std::array<double, 3>>(new_pose));


    // Add the new pose as a parameter block
    problem_.AddParameterBlock(node_poses_.back()->data(), 3);



   // Add the pose residual
   ceres::CostFunction* cost_function_pose = new RelativePoseWithBiasCostFunction(relative_pose, 1.0 / opts_.odom_pos_std, 1.0 / opts_.odom_rot_std, bias_prior, (t1-t0)*1e-6);
   problem_.AddResidualBlock(cost_function_pose, nullptr, node_poses_[node_indices_[t0]]->data(), node_poses_[node_indices_[t1]]->data(), &(*(node_biases_.back())));
   //problem_.AddResidualBlock(cost_function_pose, nullptr, node_poses_[node_indices_[t0]]->data(), node_poses_[node_indices_[t1]]->data(), &(*(node_biases_.front())));


    // Add the bias residual
    if(node_biases_.size() > 1 && opts_.estimate_bias)
    {

        ceres::CostFunction* cost_function_bias = new BrownianMotionCostFunction(1.0/ opts_.bias_std, bias_priors_[bias_priors_.size()-2], bias_priors_.back());
        problem_.AddResidualBlock(cost_function_bias, nullptr, &(*(node_biases_[node_biases_.size()-2])), &(*(node_biases_.back())));
    }

}

void PoseGraph::addLoopClosureRotEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose)
{
    int64_t local_t0 = t0;
    int64_t local_t1 = t1;
    if(node_indices_.find(t0) == node_indices_.end())
    {
        // Look for the closest timestamp to t0, if it is within 5ms use it as t0
        auto it = node_indices_.lower_bound(t0);
        if(it != node_indices_.end() && std::abs(it->first - t0) <= 5000)
        {
            local_t0 = it->first;
            std::cout << "Using closest timestamp " << local_t0 << " for loop closure edge t0 instead of " << t0 << std::endl;
        }
        else
        {
            std::cout << "No close timestamp found for loop closure edge t0 " << t0 << ", skipping this edge." << std::endl;
            return;
        }
    }
    if(node_indices_.find(t1) == node_indices_.end())
    {
        // Look for the closest timestamp to t1, if it is within 5ms use it as t1
        auto it = node_indices_.lower_bound(t1);
        if(it != node_indices_.end() && std::abs(it->first - t1) <= 5000)
        {
            local_t1 = it->first;
            std::cout << "Using closest timestamp " << local_t1 << " for loop closure edge t1 instead of " << t1 << std::endl;
        }
        else
        {
            std::cout << "No close timestamp found for loop closure edge t1 " << t1 << ", skipping this edge." << std::endl;
            return;
        }
    }

    size_t index0 = node_indices_[local_t0];
    size_t index1 = node_indices_[local_t1];

    if(index0 >= node_poses_.size() || index1 >= node_poses_.size())
    {
        throw std::runtime_error("Loop closure edge indices out of bounds.");
    }

    ceres::CostFunction* cost_function_rot = new RelativeRotCostFunction(relative_pose, 1.0 / opts_.loop_rot_std);
    problem_.AddResidualBlock(cost_function_rot, loss_function_loop_rot_, node_poses_[index0]->data(), node_poses_[index1]->data());

}

void PoseGraph::addLoopClosurePosEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose)
{
    int64_t local_t0 = t0;
    int64_t local_t1 = t1;
    if(node_indices_.find(t0) == node_indices_.end())
    {
        // Look for the closest timestamp to t0, if it is within 5ms use it as t0
        auto it = node_indices_.lower_bound(t0);
        if(it != node_indices_.end() && std::abs(it->first - t0) <= 5000)
        {
            local_t0 = it->first;
            std::cout << "Using closest timestamp " << local_t0 << " for loop closure edge t0 instead of " << t0 << std::endl;
        }
        else
        {
            std::cout << "No close timestamp found for loop closure edge t0 " << t0 << ", skipping this edge." << std::endl;
            return;
        }
    }
    if(node_indices_.find(t1) == node_indices_.end())
    {
        // Look for the closest timestamp to t1, if it is within 5ms use it as t1
        auto it = node_indices_.lower_bound(t1);
        if(it != node_indices_.end() && std::abs(it->first - t1) <= 5000)
        {
            local_t1 = it->first;
            std::cout << "Using closest timestamp " << local_t1 << " for loop closure edge t1 instead of " << t1 << std::endl;
        }
        else
        {
            std::cout << "No close timestamp found for loop closure edge t1 " << t1 << ", skipping this edge." << std::endl;
            return;
        }
    }

    size_t index0 = node_indices_[local_t0];
    size_t index1 = node_indices_[local_t1];

    if(index0 >= node_poses_.size() || index1 >= node_poses_.size())
    {
        throw std::runtime_error("Loop closure edge indices out of bounds.");
    }

    ceres::CostFunction* cost_function_pos = new RelativePosCostFunction(relative_pose, 1.0 / opts_.loop_pos_std);
    problem_.AddResidualBlock(cost_function_pos, loss_function_loop_pos_, node_poses_[index0]->data(), node_poses_[index1]->data());
}

void PoseGraph::addLoopClosureEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose)
{
    addLoopClosurePosEdge(t0, t1, relative_pose);
    addLoopClosureRotEdge(t0, t1, relative_pose);
}


void PoseGraph::optimize()
{
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 1000;
    options.num_threads = 1;
    options.function_tolerance = 1e-8;
    options.gradient_tolerance = 1e-8;
    options.parameter_tolerance = 1e-8;

    // Set all the biases equal to the median bias
    std::vector<double> biases;
    for (const auto& bias : node_biases_) {
        biases.push_back(*bias);
    }
    std::sort(biases.begin(), biases.end());
    double median_bias = biases[biases.size() / 2];
    for (auto& bias : node_biases_) {
        *bias = median_bias;
    }

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem_, &summary);
    std::cout << summary.BriefReport() << std::endl;

    // Update the loss functions with the final scale
    if ((loss_function_loop_pos_) && (first_optimization_)) {
        first_optimization_ = false;
        dynamic_cast<DynamicCauchyLoss*>(loss_function_loop_pos_)->setScale(opts_.loss_scale_loop_pos_fine / opts_.loop_pos_std);
        ceres::Solve(options, &problem_, &summary);
        std::cout << summary.BriefReport() << std::endl;
    }

}


void PoseGraph::printPoses() const
{
    for(size_t i = 0; i < node_times_.size(); ++i)
    {
        const auto& pose = *(node_poses_[i]);
        std::cout << node_times_[i] << " -> ("
                    << pose[0] << ", " << pose[1] << ", " << pose[2] << " rad)" << std::endl;
    }
}

void PoseGraph::printLastPose() const
{
    if(node_poses_.empty())
    {
        std::cout << "No poses available." << std::endl;
        return;
    }
    const auto& pose = *(node_poses_.back());
    std::cout << "Last pose: "
                << pose[0] << ", " << pose[1] << ", " << pose[2] << " rad" << std::endl;
}


void PoseGraph::writeToFile(const std::string& filename) const
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file for writing.");
    }

    for (size_t i = 0; i < node_times_.size(); ++i)
    {
        const auto& pose = *(node_poses_[i]);
        file << node_times_[i] << " "
             << pose[0] << " " << pose[1] << " " << pose[2] << "\n";
    }

    file.close();
}














RelativeRotCostFunction::RelativeRotCostFunction(const std::array<double, 3>& relative_pose, double weight)
{
    inv_meas_ = xyThetaToMat(relative_pose).inverse();
    weight_ = weight;
}

bool RelativeRotCostFunction::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
{
    const std::array<double, 3> pose1 = {parameters[0][0], parameters[0][1], parameters[0][2]};
    const std::array<double, 3> pose2 = {parameters[1][0], parameters[1][1], parameters[1][2]};

    Eigen::Matrix3d inv_mat1 = xyThetaToMat(pose1).inverse();
    Eigen::Matrix3d mat2 = xyThetaToMat(pose2);

    Eigen::Matrix3d relative_pose = inv_mat1 * mat2;
    Eigen::Matrix3d delta = inv_meas_ * relative_pose;
    residuals[0] = std::atan2(delta(1, 0), delta(0, 0)) * weight_;

    if(jacobians)
    {
        // Compute the Jacobian
        if (jacobians[0] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jacobian1(jacobians[0]);

            jacobian1.setZero();
            jacobian1(0,2) = -1;
            jacobian1 *= weight_;
        }
        if (jacobians[1] != nullptr) {
            // Jacobian w.r.t. pose2
            Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jacobian2(jacobians[1]);

            jacobian2.setZero();
            jacobian2(0,2) = 1;
            jacobian2 *= weight_;
        }
    }


    return true;
}



RelativePosCostFunction::RelativePosCostFunction(const std::array<double, 3>& relative_pose, double weight)
    : weight_(weight)
{
    meas_[0] = relative_pose[0];
    meas_[1] = relative_pose[1];

}

bool RelativePosCostFunction::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
{
    const std::array<double, 3> pose1 = {parameters[0][0], parameters[0][1], parameters[0][2]};
    const std::array<double, 3> pose2 = {parameters[1][0], parameters[1][1], parameters[1][2]};

    Eigen::Matrix3d inv_mat1 = xyThetaToMat(pose1).inverse();
    Eigen::Matrix3d mat2 = xyThetaToMat(pose2);

    Eigen::Matrix3d relative_pose = inv_mat1 * mat2;
    Eigen::Vector2d delta = relative_pose.block<2,1>(0,2) - meas_;
    residuals[0] = weight_ * delta(0);
    residuals[1] = weight_ * delta(1);

    if(jacobians)
    {
        // Compute the Jacobian
        if (jacobians[0] != nullptr)
        {
            double s1 = std::sin(pose1[2]);
            double c1 = std::cos(pose1[2]);
            double dx = pose2[0] - pose1[0];
            double dy = pose2[1] - pose1[1];

            Eigen::Vector2d temp;
            temp[0] = -s1 * dx + c1 * dy;
            temp[1] = -c1 * dx - s1 * dy;

            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian1(jacobians[0]);

            jacobian1.setZero();
            jacobian1.block<2, 2>(0, 0) = -inv_mat1.block<2, 2>(0, 0);
            jacobian1.block<2, 1>(0, 2) = temp;
            jacobian1 *= weight_;
        }
        if (jacobians[1] != nullptr)
        {
            // Jacobian w.r.t. pose2
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian2(jacobians[1]);

            jacobian2.setZero();
            jacobian2.block<2, 2>(0, 0) = inv_mat1.block<2, 2>(0, 0);
            jacobian2 *= weight_;
        }
    }


    return true;
}




RelativePoseWithBiasCostFunction::RelativePoseWithBiasCostFunction(const std::array<double, 3>& relative_pose, double pos_weight, double rot_weight, double bias_prior, double delta_t)
    : pos_weight_(pos_weight), rot_weight_(rot_weight)
{
    relative_meas_ = relative_pose;
    bias_prior_ = bias_prior;
    delta_t_ = delta_t;
}

bool RelativePoseWithBiasCostFunction::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
{
    const std::array<double, 3> pose1 = {parameters[0][0], parameters[0][1], parameters[0][2]};
    const std::array<double, 3> pose2 = {parameters[1][0], parameters[1][1], parameters[1][2]};
    const double bias_correction = parameters[2][0] - bias_prior_;

    Eigen::Matrix3d inv_mat1 = xyThetaToMat(pose1).inverse();
    Eigen::Matrix3d mat2 = xyThetaToMat(pose2);

    Eigen::Matrix3d relative_pose = inv_mat1 * mat2;
    std::array<double, 3> relative_meas_bias = relative_meas_;
    relative_meas_bias[2] -= bias_correction * delta_t_;
    Eigen::Matrix3d inv_meas = xyThetaToMat(relative_meas_bias).inverse();

    Eigen::Matrix3d delta = inv_meas * relative_pose;
    residuals[0] = pos_weight_ * (relative_pose(0, 2) - relative_meas_[0]);
    residuals[1] = pos_weight_ * (relative_pose(1, 2) - relative_meas_[1]);
    residuals[2] = rot_weight_ * std::atan2(delta(1, 0), delta(0, 0));

    if (jacobians)
    {
        // Compute the Jacobian
        if (jacobians[0] != nullptr)
        {
            double s1 = std::sin(pose1[2]);
            double c1 = std::cos(pose1[2]);
            double dx = pose2[0] - pose1[0];
            double dy = pose2[1] - pose1[1];

            Eigen::Vector2d temp;
            temp[0] = -s1 * dx + c1 * dy;
            temp[1] = -c1 * dx - s1 * dy;

            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian1(jacobians[0]);
            jacobian1.setZero();
            jacobian1.block<2, 2>(0, 0) = -inv_mat1.block<2, 2>(0, 0) * pos_weight_;
            jacobian1.block<2, 1>(0, 2) = pos_weight_ * temp;
            jacobian1(2, 2) = -rot_weight_;
        }
        if (jacobians[1] != nullptr) 
        {
            // Jacobian w.r.t. pose2
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian2(jacobians[1]);

            jacobian2.setZero();
            jacobian2.block<2, 2>(0, 0) = inv_mat1.block<2, 2>(0, 0) * pos_weight_;
            jacobian2(2, 2) = rot_weight_;
        }
        if (jacobians[2] != nullptr)
        {
            // Jacobian w.r.t. bias
            Eigen::Map<Eigen::Matrix<double, 3, 1>> jacobian3(jacobians[2]);

            jacobian3.setZero();
            jacobian3(2) = delta_t_ * rot_weight_;
        }
    }

    return true;
}











BrownianMotionCostFunction::BrownianMotionCostFunction(double weight, double bias_prior_1, double bias_prior_2)
    : weight_(weight), bias_prior_1_(bias_prior_1), bias_prior_2_(bias_prior_2)
{
}

bool BrownianMotionCostFunction::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
{
    double b1 = parameters[0][0];
    double b2 = parameters[1][0];

    residuals[0] = weight_ * (b2 - b1);

    if (jacobians)
    {
        if (jacobians[0] != nullptr)
        {
            jacobians[0][0] = -weight_;
        }
        if (jacobians[1] != nullptr)
        {
            jacobians[1][0] = weight_;
        }
    }
    return true;
}
