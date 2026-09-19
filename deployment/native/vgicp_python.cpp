#include <cmath>
#include <memory>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam_points/factors/integrated_vgicp_factor.hpp>
#include <gtsam_points/features/covariance_estimation.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

class VgicpFrame {
public:
  VgicpFrame(
      const py::array_t<double, py::array::c_style | py::array::forcecast>& values,
      int covariance_neighbors,
      int num_threads,
      double covariance_scale) {
    const auto points = values.unchecked<2>();
    if (points.shape(1) != 3 || points.shape(0) < 100) {
      throw std::invalid_argument("VGICP points must have shape (N, 3), N >= 100");
    }
    std::vector<Eigen::Vector4d> homogeneous;
    homogeneous.reserve(points.shape(0));
    for (py::ssize_t row = 0; row < points.shape(0); ++row) {
      if (std::isfinite(points(row, 0)) && std::isfinite(points(row, 1)) &&
          std::isfinite(points(row, 2))) {
        homogeneous.emplace_back(
            points(row, 0), points(row, 1), points(row, 2), 1.0);
      }
    }
    if (homogeneous.size() < 100) {
      throw std::invalid_argument("VGICP frame has fewer than 100 finite points");
    }
    cloud_ = std::make_shared<gtsam_points::PointCloudCPU>(homogeneous);
    auto covariances = gtsam_points::estimate_covariances(
        *cloud_, covariance_neighbors, num_threads);
    for (auto& covariance : covariances) {
      covariance.topLeftCorner<3, 3>() *= covariance_scale;
    }
    cloud_->add_covs(covariances);
  }

  const gtsam_points::PointCloudCPU::Ptr& cloud() const { return cloud_; }
  std::size_t size() const { return cloud_->size(); }

private:
  gtsam_points::PointCloudCPU::Ptr cloud_;
};

class VgicpTarget {
public:
  VgicpTarget(const VgicpFrame& frame, double voxel_resolution_m)
      : voxels_(std::make_shared<gtsam_points::GaussianVoxelMapCPU>(
            voxel_resolution_m)) {
    if (!(voxel_resolution_m > 0.0)) {
      throw std::invalid_argument("VGICP voxel resolution must be positive");
    }
    voxels_->insert(*frame.cloud());
  }

  const gtsam_points::GaussianVoxelMapCPU::Ptr& voxels() const { return voxels_; }

private:
  gtsam_points::GaussianVoxelMapCPU::Ptr voxels_;
};

py::dict align(
    const VgicpTarget& target,
    const VgicpFrame& source,
    const Eigen::Matrix4d& initial,
    int max_iterations,
    int num_threads) {
  if (!initial.allFinite() || max_iterations < 1) {
    throw std::invalid_argument("invalid VGICP alignment request");
  }
  py::gil_scoped_release release;
  constexpr gtsam::Key source_key = 0;
  auto factor = gtsam::make_shared<gtsam_points::IntegratedVGICPFactor>(
      gtsam::Pose3::Identity(), source_key, target.voxels(), source.cloud());
  factor->set_num_threads(num_threads);
  gtsam::NonlinearFactorGraph graph;
  graph.add(factor);
  gtsam::Values values;
  values.insert(source_key, gtsam::Pose3(initial));
  gtsam_points::LevenbergMarquardtExtParams parameters;
  parameters.setMaxIterations(max_iterations);
  parameters.setRelativeErrorTol(1.0e-5);
  parameters.setAbsoluteErrorTol(1.0e-5);
  gtsam_points::LevenbergMarquardtOptimizerExt optimizer(
      graph, values, parameters);
  const auto optimized = optimizer.optimize();
  const auto pose = optimized.at<gtsam::Pose3>(source_key);
  const double error = factor->error(optimized);
  const int inliers = factor->num_inliers();
  const double inlier_fraction = factor->inlier_fraction();
  const bool valid = pose.matrix().allFinite() && inliers > 0 &&
      std::isfinite(error) && std::isfinite(inlier_fraction);
  const double normalized_error = inliers > 0
      ? error / static_cast<double>(inliers)
      : std::numeric_limits<double>::infinity();
  py::gil_scoped_acquire acquire;
  py::dict result;
  result["pose"] = pose.matrix();
  result["valid"] = valid;
  result["num_inliers"] = inliers;
  result["inlier_fraction"] = inlier_fraction;
  result["normalized_error"] = normalized_error;
  result["iterations"] = optimizer.iterations();
  return result;
}

}  // namespace

PYBIND11_MODULE(s10_vgicp, module) {
  module.doc() = "Minimal gtsam_points VGICP binding for S10 localization";
  py::class_<VgicpFrame>(module, "Frame")
      .def(py::init<const py::array_t<double, py::array::c_style | py::array::forcecast>&,
                    int, int, double>(),
           py::arg("points"), py::arg("covariance_neighbors") = 15,
           py::arg("num_threads") = 1, py::arg("covariance_scale") = 1.0)
      .def_property_readonly("size", &VgicpFrame::size);
  py::class_<VgicpTarget>(module, "Target")
      .def(py::init<const VgicpFrame&, double>(),
           py::arg("frame"), py::arg("voxel_resolution_m"));
  module.def(
      "align", &align, py::arg("target"), py::arg("source"),
      py::arg("initial"), py::arg("max_iterations") = 10,
      py::arg("num_threads") = 1);
}
