#include "pose_graph.h"
#include "utils.h"

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <yaml-cpp/yaml.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

class PogoNode : public rclcpp::Node {
    public:
        PogoNode() : Node("pogo_node") {
            PoseGraphOpts opts;

            // Read the config file
            // Get the path to the config file from a parameter, with a default value
            std::string config_file = this->declare_parameter<std::string>("config_file", "pogo/config_temp.yaml");
            YAML::Node config = YAML::LoadFile(config_file);
            if (!config["loss_scale_loop_pos_coarse"]) {
                throw std::runtime_error("Loss scale for loop position coarse not found in config file.");
            }
            if (!config["loss_scale_loop_pos_fine"]) {
                throw std::runtime_error("Loss scale for loop position fine not found in config file.");
            }
            if (!config["loss_scale_loop_rot"]) {
                throw std::runtime_error("Loss scale for loop rotation not found in config file.");
            }
            if (!config["odom_pos_std"]) {
                throw std::runtime_error("Odometry position standard deviation not found in config file.");
            }
            if (!config["odom_rot_std"]) {
                throw std::runtime_error("Odometry rotation standard deviation not found in config file.");
            }
            if (!config["loop_pos_std"]) {
                throw std::runtime_error("Loop position standard deviation not found in config file.");
            }
            if (!config["loop_rot_std"]) {
                throw std::runtime_error("Loop rotation standard deviation not found in config file.");
            }
            if (!config["estimate_bias"]) {
                throw std::runtime_error("Estimate bias not found in config file.");
            }
            else
            {
                opts.estimate_bias = config["estimate_bias"].as<bool>();
                if(!config["bias_walk_std"]){
                    throw std::runtime_error("Bias walk standard deviation not found in config file.");
                }
                opts.bias_std = config["bias_walk_std"].as<double>();
            }
            if (!config["output_traj_path"]) {
                output_traj_path_ = "pose_graph_traj.txt";
            }
            else
            {
                output_traj_path_ = config["output_traj_path"].as<std::string>();
            }


            opts.loss_scale_loop_pos_coarse = config["loss_scale_loop_pos_coarse"].as<double>();
            opts.loss_scale_loop_pos_fine = config["loss_scale_loop_pos_fine"].as<double>();
            opts.loss_scale_loop_rot = config["loss_scale_loop_rot"].as<double>() * M_PI / 180.0;
            opts.odom_pos_std = config["odom_pos_std"].as<double>();
            opts.odom_rot_std = config["odom_rot_std"].as<double>() * M_PI / 180.0;
            opts.loop_pos_std = config["loop_pos_std"].as<double>();
            opts.loop_rot_std = config["loop_rot_std"].as<double>() * M_PI / 180.0;


            odom_topic_ = "/dro_odometry";
            loop_topic_ = "/registration_relative_pose";

            pose_graph_ = std::make_unique<PoseGraph>(opts);

            odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
                odom_topic_,
                rclcpp::SystemDefaultsQoS(),
                std::bind(&PogoNode::onOdometry, this, std::placeholders::_1)
            );

            loop_sub_ = this->create_subscription<geometry_msgs::msg::TransformStamped>(
                loop_topic_,
                rclcpp::SystemDefaultsQoS(),
                std::bind(&PogoNode::onLoopTransform, this, std::placeholders::_1)
            );

            RCLCPP_INFO(get_logger(), "Started pogo_node. Subscribed to odom='%s', loop_transform='%s'.",
                        odom_topic_.c_str(), loop_topic_.c_str());
            RCLCPP_INFO(get_logger(), "Loop edges use frame IDs as timestamps: header.frame_id -> t0, child_frame_id -> t1.");
        }


    private:
        static int64_t stampToNanoseconds(const builtin_interfaces::msg::Time & stamp) {
            return static_cast<int64_t>(stamp.sec) * 1000000000LL + static_cast<int64_t>(stamp.nanosec);
        }

        static bool parseInt64(const std::string & text, int64_t & out_value) {
            if (text.empty()) {
                return false;
            }
            try {
                std::size_t parsed = 0;
                out_value = std::stoll(text, &parsed);
                return parsed == text.size();
            } catch (const std::exception &) {
                return false;
            }
        }

        static double yawFromQuaternion(double x, double y, double z, double w) {
            const double siny_cosp = 2.0 * (w * z + x * y);
            const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
            return std::atan2(siny_cosp, cosy_cosp);
        }

        void writeTrajectoryToFile() const {
            std::ofstream file(output_traj_path_);
            if (!file.is_open()) {
                RCLCPP_ERROR(get_logger(), "Failed to open trajectory output file: %s", output_traj_path_.c_str());
                return;
            }
            for (size_t i = 0; i < pose_graph_->node_times_.size(); ++i) {
                const auto & t = pose_graph_->node_times_[i];
                const auto & pose = *pose_graph_->node_poses_[i];
                file << t << " " << pose[0] << " " << pose[1] << " " << pose[2] << "\n";
            }
            file.close();
            RCLCPP_INFO(get_logger(), "Trajectory written to: %s", output_traj_path_.c_str());
        }

        void onOdometry(const nav_msgs::msg::Odometry::SharedPtr msg) {
            const int64_t t_curr = stampToNanoseconds(msg->header.stamp);

            std::array<double, 3> curr_pose{
                msg->pose.pose.position.x,
                msg->pose.pose.position.y,
                yawFromQuaternion(
                    msg->pose.pose.orientation.x,
                    msg->pose.pose.orientation.y,
                    msg->pose.pose.orientation.z,
                    msg->pose.pose.orientation.w)
            };

            if (!last_odom_pose_.has_value()) {
                last_odom_time_ = t_curr;
                last_odom_pose_ = curr_pose;
                return;
            }

            const auto relative_pose = relativePose(last_odom_pose_.value(), curr_pose);
            pose_graph_->addOdometryEdge(last_odom_time_, t_curr, relative_pose, default_bias_prior_);

            last_odom_time_ = t_curr;
            last_odom_pose_ = curr_pose;

            writeTrajectoryToFile();
        }

        void onLoopTransform(const geometry_msgs::msg::TransformStamped::SharedPtr msg) {
            int64_t t0 = 0;
            int64_t t1 = 0;
            const bool ok_t0 = parseInt64(msg->header.frame_id, t0);
            const bool ok_t1 = parseInt64(msg->child_frame_id, t1);
            if (!ok_t0 || !ok_t1) {
                RCLCPP_WARN_THROTTLE(
                    get_logger(),
                    *get_clock(),
                    5000,
                    "Skipped loop transform: frame IDs must be integer timestamps (frame_id='%s', child_frame_id='%s').",
                    msg->header.frame_id.c_str(),
                    msg->child_frame_id.c_str());
                return;
            }

            const auto & t = msg->transform.translation;
            const auto & q = msg->transform.rotation;
            const std::array<double, 3> relative_pose{
                t.x,
                t.y,
                yawFromQuaternion(q.x, q.y, q.z, q.w)
            };

            pose_graph_->addLoopClosureEdge(t0, t1, relative_pose);
            pose_graph_->optimize();
            pose_graph_->printLastPose();
            writeTrajectoryToFile();
        }

        std::unique_ptr<PoseGraph> pose_graph_;

        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
        rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr loop_sub_;

        std::string odom_topic_;
        std::string loop_topic_;
        bool save_traj_on_shutdown_ = false;
        std::string output_traj_path_;
        double default_bias_prior_ = 0.0;

        int64_t last_odom_time_ = 0;
        std::optional<std::array<double, 3>> last_odom_pose_;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PogoNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

