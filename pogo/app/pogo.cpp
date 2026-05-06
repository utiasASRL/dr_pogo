#include "pose_graph.h"
#include "utils.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <chrono>


bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}


int main()
{
    //auto seq_id = getSeqId();
    //std::cout << "Sequence ID: " << seq_id << std::endl;

    PoseGraphOpts opts;

    // Read the config file
    YAML::Node config = YAML::LoadFile("pogo/config_temp.yaml");
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
    bool use_coarse_registration = false;
    if (config["use_coarse_registration"]) {
        use_coarse_registration = config["use_coarse_registration"].as<bool>();
    }
    if (!config["seq_id"]) {
        throw std::runtime_error("Seq id not found in the config file");
    }


    opts.loss_scale_loop_pos_coarse = config["loss_scale_loop_pos_coarse"].as<double>();
    opts.loss_scale_loop_pos_fine = config["loss_scale_loop_pos_fine"].as<double>();
    opts.loss_scale_loop_rot = config["loss_scale_loop_rot"].as<double>() * M_PI / 180.0;
    opts.odom_pos_std = config["odom_pos_std"].as<double>();
    opts.odom_rot_std = config["odom_rot_std"].as<double>() * M_PI / 180.0;
    opts.loop_pos_std = config["loop_pos_std"].as<double>();
    opts.loop_rot_std = config["loop_rot_std"].as<double>() * M_PI / 180.0;

    std::string seq_id = config["seq_id"].as<std::string>();

    // Initialize the pose graph
    PoseGraph pose_graph(opts);

    // Read the odometry data
    std::string odometry_file = "output/" + seq_id + "/odometry_2d/" + seq_id + ".txt";
    std::cout << "Reading odometry data from: " << odometry_file << std::endl;
    std::vector<std::pair<int64_t, std::array<double, 3>>> odometry_data = readOdometry(odometry_file);

    std::vector<double> bias_priors;
    std::string bias_file = "output/" + seq_id + "/other_log/gyro_bias.txt";
    if(opts.estimate_bias && fileExists(bias_file))
    {
        bias_priors = readBiases(bias_file);
    }
    else
    {
        bias_priors = std::vector<double>(odometry_data.size() - 1, 0.0);
    }

    std::cout << "Read " << odometry_data.size() << " odometry entries." << std::endl;
    std::cout << "Read " << bias_priors.size() << " bias entries." << std::endl;

    // Read the loop closure data
    std::string loop_closure_file = "output/" + seq_id + "/";
    if (use_coarse_registration) {
        loop_closure_file += "coarse_registrations.csv";
    } else {
        loop_closure_file += "fine_registrations.csv";
    }
    std::cout << "Reading loop closure data from: " << loop_closure_file << std::endl;
    std::vector<std::tuple<int64_t, int64_t, std::array<double, 3>>> loop_closures = readLoopClosures(loop_closure_file);


    // Record time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Add odometry edges to the pose graph
    for(size_t i = 0; i < odometry_data.size() - 1; ++i) {
        const auto& [t0, pose] = odometry_data[i];
        const auto& [t1, next_pose] = odometry_data[i + 1];

        const auto relative_pose = relativePose(pose, next_pose);
        pose_graph.addOdometryEdge(t0, t1, relative_pose, bias_priors[i]);
    }


    pose_graph.printLastPose();


    // Add loop closure edges to the pose graph
    for(const auto& [t0, t1, relative_pose] : loop_closures) {
        pose_graph.addLoopClosureEdge(t0, t1, relative_pose);
        pose_graph.optimize();
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    // Write optimization time to file
    std::chrono::duration<double> optimization_duration = end_time - start_time;
    std::cout << "Optimization took " << optimization_duration.count() << " seconds." << std::endl;
    std::ofstream time_file("output/" + seq_id + "/pogo_time.txt");
    if (time_file.is_open()) {
        time_file << optimization_duration.count() << std::endl;
        time_file.close();
    } else {
        std::cerr << "Unable to open file to write optimization time." << std::endl;
    }

    pose_graph.printLastPose();

    pose_graph.writeToFile("output/" + seq_id + "/pose_graph_traj.txt");

    return 0;
}

