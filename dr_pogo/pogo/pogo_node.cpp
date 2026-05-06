#include "pose_graph.h"
#include "utils.h"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/wait_for_message.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <yaml-cpp/yaml.h>
#include <dr_pogo/msg/radar_info.hpp>


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
            // Get the path to the config file from a parameter without a default value (must be provided by the user)
            std::string config_file;
            this->declare_parameter<std::string>("config_file", "");
            if (!this->get_parameter("config_file").as_string().empty()) {
                config_file = this->get_parameter("config_file").as_string();
            } else {
                throw std::runtime_error("Config file path must be provided as a parameter 'config_file'.");
            }

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


            opts.loss_scale_loop_pos_coarse = config["loss_scale_loop_pos_coarse"].as<double>();
            opts.loss_scale_loop_pos_fine = config["loss_scale_loop_pos_fine"].as<double>();
            opts.loss_scale_loop_rot = config["loss_scale_loop_rot"].as<double>() * M_PI / 180.0;
            opts.odom_pos_std = config["odom_pos_std"].as<double>();
            opts.odom_rot_std = config["odom_rot_std"].as<double>() * M_PI / 180.0;
            opts.loop_pos_std = config["loop_pos_std"].as<double>();
            opts.loop_rot_std = config["loop_rot_std"].as<double>() * M_PI / 180.0;


            odom_topic_ = "/dro_local_map_odometry";
            loop_topic_ = "/registration_relative_pose";

            pose_graph_ = std::make_unique<PoseGraph>(opts);

            // Subscribe with a stack size of 100 messages to avoid missing messages in case of slow processing
            odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
                odom_topic_,
                rclcpp::SystemDefaultsQoS().keep_last(100),
                std::bind(&PogoNode::onOdometry, this, std::placeholders::_1)
            );

            loop_sub_ = this->create_subscription<geometry_msgs::msg::TransformStamped>(
                loop_topic_,
                rclcpp::SystemDefaultsQoS(),
                std::bind(&PogoNode::onLoopTransform, this, std::placeholders::_1)
            );

            path_pub_ = this->create_publisher<nav_msgs::msg::Path>("pogo_path", rclcpp::SystemDefaultsQoS());



            RCLCPP_INFO(get_logger(), "Started pogo_node. Subscribed to odom='%s', loop_transform='%s'.",
                        odom_topic_.c_str(), loop_topic_.c_str());

            // Read the output path base from the the parameter "output_path"
            this->declare_parameter<std::string>("output_path", "");
            output_traj_path_ = this->get_parameter("output_path").as_string();
            if(output_traj_path_.size() > 0 && output_traj_path_.back() != '/')
            {
                output_traj_path_ += '/';
            }
            RCLCPP_INFO(get_logger(), "Waiting for a single radar info message to initialize the trajectory output file with the sequence ID");
            dr_pogo::msg::RadarInfo radar_info_msg;
            auto node_ptr = std::shared_ptr<rclcpp::Node>(this, [](rclcpp::Node *) {});
            if (rclcpp::wait_for_message(
                    radar_info_msg,
                    node_ptr,
                    "/boreas/radar_info"))
            {
                std::string sequence_id = radar_info_msg.sequence_id;
                output_traj_path_ = output_traj_path_ + sequence_id + "/pose_graph_traj.txt";
                RCLCPP_INFO(get_logger(), "Got radar info, sequence_id='%s'", sequence_id.c_str());
            } else {
                throw std::runtime_error("Timed out waiting for radar_info message.");
            }

            
        }


    private:
        static int64_t stampToMicroseconds(const builtin_interfaces::msg::Time & stamp) {
            return static_cast<int64_t>(stamp.sec) * 1000000LL + static_cast<int64_t>(stamp.nanosec) / 1000LL;
        }

        static builtin_interfaces::msg::Time microsecondsToStamp(int64_t microseconds) {
            builtin_interfaces::msg::Time time;
            time.sec = microseconds / 1000000LL;
            time.nanosec = (microseconds % 1000000LL) * 1000LL;
            return time;
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

        void writeTrajAndPublish() const {
            // Measure the time taken by writing the trajectory to file and publishing the path
            pose_graph_->writeToFile(output_traj_path_);
            if (path_pub_) {
                auto [times, poses] = pose_graph_->getPoses();

                nav_msgs::msg::Path path_msg;
                path_msg.header.stamp = microsecondsToStamp(times.back());
                path_msg.header.frame_id = "map";
                for (size_t i = 0; i < poses.size(); ++i) {
                    geometry_msgs::msg::PoseStamped pose_stamped;
                    pose_stamped.header.stamp = microsecondsToStamp(times[i]);
                    pose_stamped.header.frame_id = "radar";
                    pose_stamped.pose.position.x = poses[i][0];
                    pose_stamped.pose.position.y = poses[i][1];
                    Eigen::Quaterniond q = yawToQuaternion(poses[i][2]);
                    pose_stamped.pose.orientation.x = q.x();
                    pose_stamped.pose.orientation.y = q.y();
                    pose_stamped.pose.orientation.z = q.z();
                    pose_stamped.pose.orientation.w = q.w();
                    path_msg.poses.push_back(pose_stamped);
                }
                path_pub_->publish(path_msg);
            }
        }

        void onOdometry(const nav_msgs::msg::Odometry::SharedPtr msg) {
            const int64_t t_curr = stampToMicroseconds(msg->header.stamp);
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

            writeTrajAndPublish();
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
            writeTrajAndPublish();
        }

        std::unique_ptr<PoseGraph> pose_graph_;

        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
        rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr loop_sub_;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

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

