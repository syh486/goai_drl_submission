#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {

gtsam::Key key_for(std::size_t index) {
  return gtsam::Symbol('x', index);
}

gtsam::Pose3 parse_pose(const json& value) {
  if (!value.is_array() || value.size() != 4) {
    throw std::runtime_error("pose must be a 4x4 array");
  }
  gtsam::Matrix3 rotation;
  gtsam::Point3 translation;
  for (int row = 0; row < 3; ++row) {
    if (!value[row].is_array() || value[row].size() != 4) {
      throw std::runtime_error("pose row must contain four values");
    }
    for (int column = 0; column < 3; ++column) {
      rotation(row, column) = value[row][column].get<double>();
    }
    translation(row) = value[row][3].get<double>();
  }
  return gtsam::Pose3(gtsam::Rot3(rotation), translation);
}

json serialize_pose(const gtsam::Pose3& pose) {
  const auto matrix = pose.matrix();
  json value = json::array();
  for (int row = 0; row < 4; ++row) {
    json output_row = json::array();
    for (int column = 0; column < 4; ++column) {
      output_row.push_back(matrix(row, column));
    }
    value.push_back(std::move(output_row));
  }
  return value;
}

gtsam::SharedNoiseModel noise_model(
    double translation_sigma,
    double rotation_sigma,
    bool robust,
    double huber_width) {
  gtsam::Vector6 sigmas;
  sigmas << rotation_sigma, rotation_sigma, rotation_sigma,
      translation_sigma, translation_sigma, translation_sigma;
  auto diagonal = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
  if (!robust) {
    return diagonal;
  }
  return gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Huber::Create(huber_width), diagonal);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "Usage: multilap_pose_graph INPUT.json OUTPUT.json\n";
      return 2;
    }
    std::ifstream input_stream(argv[1]);
    if (!input_stream) {
      throw std::runtime_error("failed to open input JSON");
    }
    json input;
    input_stream >> input;
    const auto& nodes = input.at("nodes");
    if (!nodes.is_array() || nodes.size() < 2) {
      throw std::runtime_error("at least two nodes are required");
    }

    gtsam::Values initial;
    for (std::size_t index = 0; index < nodes.size(); ++index) {
      initial.insert(key_for(index), parse_pose(nodes[index]));
    }
    gtsam::NonlinearFactorGraph graph;
    const std::size_t fixed_node = input.value("fixed_node", 0U);
    gtsam::Vector6 prior_sigmas;
    prior_sigmas << 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6;
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        key_for(fixed_node), initial.at<gtsam::Pose3>(key_for(fixed_node)),
        gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas));

    for (const auto& factor : input.at("factors")) {
      const std::size_t first = factor.at("first").get<std::size_t>();
      const std::size_t second = factor.at("second").get<std::size_t>();
      if (first >= nodes.size() || second >= nodes.size() || first == second) {
        throw std::runtime_error("factor references invalid nodes");
      }
      graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
          key_for(first), key_for(second), parse_pose(factor.at("measurement")),
          noise_model(
              factor.at("translation_sigma_m").get<double>(),
              factor.at("rotation_sigma_rad").get<double>(),
              factor.value("robust", false),
              factor.value("huber_width", 1.345)));
    }

    gtsam::LevenbergMarquardtParams parameters;
    parameters.setVerbosityLM("SILENT");
    parameters.setMaxIterations(input.value("max_iterations", 200));
    parameters.setRelativeErrorTol(input.value("relative_error_tolerance", 1.0e-7));
    parameters.setAbsoluteErrorTol(input.value("absolute_error_tolerance", 1.0e-7));
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, parameters);
    const double initial_error = graph.error(initial);
    const gtsam::Values result = optimizer.optimize();
    const double final_error = graph.error(result);

    json output;
    output["success"] = std::isfinite(final_error) && final_error < initial_error;
    output["initial_error"] = initial_error;
    output["final_error"] = final_error;
    output["iterations"] = optimizer.iterations();
    output["poses"] = json::array();
    for (std::size_t index = 0; index < nodes.size(); ++index) {
      output["poses"].push_back(
          serialize_pose(result.at<gtsam::Pose3>(key_for(index))));
    }
    std::ofstream output_stream(argv[2]);
    if (!output_stream) {
      throw std::runtime_error("failed to open output JSON");
    }
    output_stream << output.dump(2) << '\n';
    return output["success"].get<bool>() ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "multilap_pose_graph: " << error.what() << '\n';
    return 2;
  }
}
