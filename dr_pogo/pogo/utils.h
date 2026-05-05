#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <fstream>
#include <vector>
#include <array>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <string>



inline Eigen::Matrix3d xyThetaToMat(const std::array<double, 3>& pose)
{
    Eigen::Matrix3d mat = Eigen::Matrix3d::Identity();
    mat(0, 0) = std::cos(pose[2]);
    mat(0, 1) = -std::sin(pose[2]);
    mat(1, 0) = std::sin(pose[2]);
    mat(1, 1) = std::cos(pose[2]);
    mat(0, 2) = pose[0];
    mat(1, 2) = pose[1];
    return mat;
}

inline std::array<double, 3> matToXYTheta(const Eigen::Matrix3d& mat)
{
    std::array<double, 3> pose;
    pose[0] = mat(0, 2);
    pose[1] = mat(1, 2);
    pose[2] = std::atan2(mat(1, 0), mat(0, 0));
    return pose;
}

inline std::array<double, 3> relativePose(
    const std::array<double, 3>& pose1,
    const std::array<double, 3>& pose2
    )
{
    Eigen::Matrix3d mat1 = xyThetaToMat(pose1);
    Eigen::Matrix3d mat2 = xyThetaToMat(pose2);
    Eigen::Matrix3d relative_mat = mat1.inverse() * mat2;
    return matToXYTheta(relative_mat);
}


inline std::array<double, 3> combinePoses(
    const std::array<double, 3>& pose1,
    const std::array<double, 3>& pose2
    )
{
    Eigen::Matrix3d mat1 = xyThetaToMat(pose1);
    Eigen::Matrix3d mat2 = xyThetaToMat(pose2);
    Eigen::Matrix3d combined_mat = mat1 * mat2;
    return matToXYTheta(combined_mat);
}
    
inline Eigen::Quaterniond yawToQuaternion(double yaw)
{
    return Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
}



// Read the csv file containing odometry data (timestamp, x, y, theta)
inline std::vector<std::pair<int64_t, std::array<double, 3>>> readOdometry(const std::string& file_path)
{
    std::vector<std::pair<int64_t, std::array<double, 3>>> odometry_data;
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open odometry file: " + file_path);
    }

    int64_t timestamp;
    double x, y, theta;
    while (file >> timestamp >> x >> y >> theta) {
        odometry_data.emplace_back(timestamp, std::array<double, 3>{x, y, theta});
    }
    file.close();
    return odometry_data;
}

inline std::vector<double> readBiases(const std::string& file_path)
{
    std::vector<double> biases;
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open biases file: " + file_path);
    }

    double bias;
    while (file >> bias) {
        biases.push_back(bias);
    }
    file.close();
    return biases;
}

inline int64_t fileNameToTimestamp(const std::string& file_name)
{
    // Extract the timestamp from the file name
    size_t pos = file_name.find_first_of(".");
    if (pos == std::string::npos) {
        throw std::runtime_error("Invalid file name format: " + file_name);
    }
    std::string timestamp_str = file_name.substr(0, pos);
    try {
        return std::stoll(timestamp_str);
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error("Invalid timestamp in file name: " + file_name);
    }
}

inline std::vector<std::tuple<int64_t, int64_t, std::array<double, 3>>> readLoopClosures(const std::string& file_path)
{
    std::vector<std::tuple<int64_t, int64_t, std::array<double, 3>>> loop_closures;
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open loop closure file: " + file_path);
    }

    std::string line;
    std::getline(file, line); // Skip header line
    while (std::getline(file, line))
    {
        std::istringstream ss(line);
        // Split the line by commas
        std::string token;
        int64_t t0, t1;
        double x, y, theta;
        if (std::getline(ss, token, ',') && !token.empty())
        {
            t0 = fileNameToTimestamp(token);
        }
        else
        {
            throw std::runtime_error("Invalid or missing timestamp t0 in loop closure data.");
        }
        if (std::getline(ss, token, ',') && !token.empty())
        {
            t1 = fileNameToTimestamp(token);
        }
        else
        {
            throw std::runtime_error("Invalid or missing timestamp t1 in loop closure data.");
        }
        if (std::getline(ss, token, ',') && !token.empty())
        {
            x = std::stod(token);
        }
        else
        {
            throw std::runtime_error("Invalid or missing x coordinate in loop closure data.");
        }
        if (std::getline(ss, token, ',') && !token.empty())
        {
            y = std::stod(token);
        }
        else
        {            throw std::runtime_error("Invalid or missing y coordinate in loop closure data.");
        }
        if (std::getline(ss, token, ',') && !token.empty())
        {
            theta = std::stod(token);
        }
        else
        {
            throw std::runtime_error("Invalid or missing theta in loop closure data.");
        }
        loop_closures.emplace_back(t0, t1, std::array<double, 3>{x, y, theta});
    }
    if (loop_closures.empty()) {
        std::cout << "No loop closures found in file: " << file_path << std::endl;
    }
    file.close();
    return loop_closures;
}



// Function to get the sequence ID from the DRO config file
inline std::string getSeqId()
{
    // Read the DRO config file
    YAML::Node config = YAML::LoadFile("dro/config.yaml");
    if (!config["data"]["data_path"]) {
        throw std::runtime_error("Data path not found in config file.");
    }
    std::string data_path = config["data"]["data_path"].as<std::string>();

    // If the path ends with a slash, remove it
    if (data_path.back() == '/')
    {
        data_path.pop_back();
    }

    // Extract the sequence ID from the data path
    std::string seq_id = data_path.substr(data_path.find_last_of("/") + 1);
    return seq_id;
}

