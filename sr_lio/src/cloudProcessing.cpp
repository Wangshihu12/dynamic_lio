#include "cloudProcessing.h"
#include "utility.h"

class PlaneFactor : public ceres::SizedCostFunction<1, 4>
{
public:
     PlaneFactor(double x, double y, double z)
         : x_(x), y_(y), z_(z) {}

     virtual ~PlaneFactor() {};

     virtual bool Evaluate(
         double const *const *parameters, double *residuals, double **jacobians) const
     {
          double a = parameters[0][0];
          double b = parameters[0][1];
          double c = parameters[0][2];
          double d = parameters[0][3];

          double n = a * x_ + b * y_ + c * z_ + d;
          double m = sqrt(a * a + b * b + c * c);
          residuals[0] = n / m;

          if (jacobians != NULL && jacobians[0] != NULL)
          {
               jacobians[0][0] = (x_ * m - a / m * n) / (m * m);
               jacobians[0][1] = (y_ * m - b / m * n) / (m * m);
               jacobians[0][2] = (z_ * m - c / m * n) / (m * m);
               jacobians[0][3] = 1 / m;
          }
          return true;
     }

private:
     const double x_;
     const double y_;
     const double z_;
};

cloudProcessing::cloudProcessing()
{
     point_filter_num = 1;
     sweep_id = 0;
}

void cloudProcessing::initialization()
{
     full_cloud.resize(N_SCANS * Horizon_SCANS);

     nan_point.raw_point.x() = std::numeric_limits<float>::quiet_NaN();
     nan_point.raw_point.y() = std::numeric_limits<float>::quiet_NaN();
     nan_point.raw_point.z() = std::numeric_limits<float>::quiet_NaN();
     nan_point.point.x() = std::numeric_limits<float>::quiet_NaN();
     nan_point.point.y() = std::numeric_limits<float>::quiet_NaN();
     nan_point.point.z() = std::numeric_limits<float>::quiet_NaN();
     nan_point.label = -1;

     sensor_mount_angle = 0.0;
     segment_theta = 60.0 / 180.0 * M_PI;
     segment_valid_point_num = 5;
     segment_valid_line_num = 3;
     segment_alpha_x = ang_res_x / 180.0 * M_PI;
     segment_alpha_y = ang_res_y / 180.0 * M_PI;

     std::pair<int8_t, int8_t> neighbor;
     neighbor.first = -1;
     neighbor.second = 0;
     neighbor_iterator.push_back(neighbor);
     neighbor.first = 0;
     neighbor.second = 1;
     neighbor_iterator.push_back(neighbor);
     neighbor.first = 0;
     neighbor.second = -1;
     neighbor_iterator.push_back(neighbor);
     neighbor.first = 1;
     neighbor.second = 0;
     neighbor_iterator.push_back(neighbor);

     all_pushed_id_x = new uint16_t[N_SCANS * Horizon_SCANS];
     all_pushed_id_y = new uint16_t[N_SCANS * Horizon_SCANS];

     queue_id_x = new uint16_t[N_SCANS * Horizon_SCANS];
     queue_id_y = new uint16_t[N_SCANS * Horizon_SCANS];

     resetParameters();
}

void cloudProcessing::resetParameters()
{
     range_mat = cv::Mat(N_SCANS, Horizon_SCANS, CV_32F, cv::Scalar::all(FLT_MAX));
     ground_mat = cv::Mat(N_SCANS, Horizon_SCANS, CV_8S, cv::Scalar::all(0));
     label_mat = cv::Mat(N_SCANS, Horizon_SCANS, CV_32S, cv::Scalar::all(0));
     label_count = 1;

     std::fill(full_cloud.begin(), full_cloud.end(), nan_point);
}

void cloudProcessing::setLidarType(int para)
{
     lidar_type = para;
}

void cloudProcessing::setNumScans(int para)
{
     N_SCANS = para;

     for (int i = 0; i < N_SCANS; i++)
     {
          pcl::PointCloud<pcl::PointXYZINormal> v_cloud_temp;
          v_cloud_temp.clear();
          scan_cloud.push_back(v_cloud_temp);
     }

     assert(N_SCANS == scan_cloud.size());

     for (int i = 0; i < N_SCANS; i++)
     {
          std::vector<extraElement> v_elem_temp;
          v_extra_elem.push_back(v_elem_temp);
     }

     assert(N_SCANS == v_extra_elem.size());
}

void cloudProcessing::setHorizonScans(int para)
{
     Horizon_SCANS = para;
}

void cloudProcessing::setAngResX(float para)
{
     ang_res_x = para;
}

void cloudProcessing::setAngResY(float para)
{
     ang_res_y = para;
}

void cloudProcessing::setAngBottom(float para)
{
     ang_bottom = para;
}

void cloudProcessing::setGroundScanInd(int para)
{
     ground_scan_id = para;
}

void cloudProcessing::setScanRate(int para)
{
     SCAN_RATE = para;
     delta_cut_time = 1.0 / (double)SCAN_RATE * 1000.0;
}

void cloudProcessing::setTimeUnit(int para)
{
     time_unit = para;

     switch (time_unit)
     {
     case SEC:
          time_unit_scale = 1.e3f;
          break;
     case MS:
          time_unit_scale = 1.f;
          break;
     case US:
          time_unit_scale = 1.e-3f;
          break;
     case NS:
          time_unit_scale = 1.e-6f;
          break;
     default:
          time_unit_scale = 1.f;
          break;
     }
}

void cloudProcessing::setBlind(double para)
{
     blind = para;
}

void cloudProcessing::setExtrinR(Eigen::Matrix3d &R)
{
     R_imu_lidar = R;
}

void cloudProcessing::setExtrinT(Eigen::Vector3d &t)
{
     t_imu_lidar = t;
}

void cloudProcessing::setPointFilterNum(int para)
{
     point_filter_num = para;
}

void cloudProcessing::process(const sensor_msgs::PointCloud2::ConstPtr &msg, std::vector<point3D> &v_cloud_out, double &dt_offset)
{
     switch (lidar_type)
     {
     case OUST:
          ousterHandler(msg, dt_offset);
          break;

     case VELO:
          velodyneHandler(msg, dt_offset);
          break;

     case ROBO:
          robosenseHandler(msg, dt_offset);
          break;

     default:
          ROS_ERROR("Only Velodyne LiDAR interface is supported currently.");
          break;
     }
     groundRemoval();

     cloudSegmentation(v_cloud_out);

     sweep_id++;

     resetParameters();
}

void cloudProcessing::ousterHandler(const sensor_msgs::PointCloud2::ConstPtr &msg, double &dt_offset)
{
     pcl::PointCloud<ouster_ros::Point> raw_cloud;
     pcl::fromROSMsg(*msg, raw_cloud);
     int size = raw_cloud.points.size();

     double dt_last_point;

     if (size == 0)
     {
          dt_offset = delta_cut_time;
          return;
     }

     if (raw_cloud.points[size - 1].t > 0)
          given_offset_time = true;
     else
          given_offset_time = false;

     if (raw_cloud.points[size - 1].ring > 0)
          given_ring = true;
     else
          given_ring = false;

     if (given_offset_time)
     {
          sort(raw_cloud.points.begin(), raw_cloud.points.end(), time_list_ouster);
          dt_last_point = raw_cloud.points.back().t * time_unit_scale;
          delta_cut_time = dt_last_point;
     }

     double omega = 0.361 * SCAN_RATE;

     std::vector<bool> is_first;
     is_first.resize(N_SCANS);
     fill(is_first.begin(), is_first.end(), true);

     std::vector<double> yaw_first_point;
     yaw_first_point.resize(N_SCANS);
     fill(yaw_first_point.begin(), yaw_first_point.end(), 0.0);

     double max_relative_time = -1.0;
     int num_full_points = 0;

     for (int i = 0; i < size; i++)
     {
          if (std::isnan(raw_cloud.points[i].x) || std::isnan(raw_cloud.points[i].y) || std::isnan(raw_cloud.points[i].z))
               continue;

          point3D point_temp;

          point_temp.raw_point = Eigen::Vector3d(raw_cloud.points[i].x, raw_cloud.points[i].y, raw_cloud.points[i].z);
          point_temp.point = point_temp.raw_point;
          point_temp.relative_time = raw_cloud.points[i].t * time_unit_scale;

          int row_id, col_id;
          double vertical_angle, horizon_angle, yaw_angle, range;

          if (given_ring)
               row_id = raw_cloud.points[i].ring;
          else
          {
               vertical_angle = atan2(point_temp.raw_point.z(), sqrt(point_temp.raw_point.x() * point_temp.raw_point.x() + point_temp.raw_point.y() * point_temp.raw_point.y())) * 180 / M_PI;

               row_id = (vertical_angle + ang_bottom) / ang_res_y;
               if (row_id < 0 || row_id >= N_SCANS)
                    continue;
          }

          yaw_angle = atan2(point_temp.raw_point.y(), point_temp.raw_point.x()) * 57.2957;

          if (is_first[row_id])
          {
               horizon_angle = atan2(point_temp.raw_point.x(), point_temp.raw_point.y()) * 180 / M_PI;

               col_id = -round((horizon_angle - 90.0) / ang_res_x) + Horizon_SCANS / 2;
               if (col_id >= Horizon_SCANS)
                    col_id -= Horizon_SCANS;

               if (col_id < 0 || col_id >= Horizon_SCANS)
                    continue;

               range = sqrt(point_temp.raw_point.x() * point_temp.raw_point.x() + point_temp.raw_point.y() * point_temp.raw_point.y() + point_temp.raw_point.z() * point_temp.raw_point.z());

               if (range < blind)
                    continue;

               point_temp.range = range;

               yaw_first_point[row_id] = yaw_angle;
               is_first[row_id] = false;

               if (given_offset_time)
               {
                    point_temp.timestamp = point_temp.relative_time / double(1000) + msg->header.stamp.toSec();
                    point_temp.alpha_time = point_temp.relative_time / dt_last_point;
               }
               else
                    point_temp.relative_time = 0.0;

               range_mat.at<float>(row_id, col_id) = range;

               num_full_points++;

               full_cloud[col_id + row_id * Horizon_SCANS] = point_temp;

               continue;
          }

          if (given_offset_time)
          {
               point_temp.timestamp = point_temp.relative_time / double(1000) + msg->header.stamp.toSec();
               point_temp.alpha_time = point_temp.relative_time / dt_last_point;
          }
          else
          {
               if (yaw_angle <= yaw_first_point[row_id])
                    point_temp.relative_time = (yaw_first_point[row_id] - yaw_angle) / omega;
               else
                    point_temp.relative_time = (yaw_first_point[row_id] - yaw_angle + 360.0) / omega;

               point_temp.timestamp = point_temp.relative_time / double(1000) + msg->header.stamp.toSec();

               if (point_temp.relative_time > max_relative_time)
                    max_relative_time = point_temp.relative_time;
          }

          horizon_angle = atan2(point_temp.raw_point.x(), point_temp.raw_point.y()) * 180 / M_PI;

          col_id = -round((horizon_angle - 90.0) / ang_res_x) + Horizon_SCANS / 2;
          if (col_id >= Horizon_SCANS)
               col_id -= Horizon_SCANS;

          if (col_id < 0 || col_id >= Horizon_SCANS)
               continue;

          range = sqrt(point_temp.raw_point.x() * point_temp.raw_point.x() + point_temp.raw_point.y() * point_temp.raw_point.y() + point_temp.raw_point.z() * point_temp.raw_point.z());

          if (range < blind)
               continue;

          point_temp.range = range;

          range_mat.at<float>(row_id, col_id) = range;

          num_full_points++;

          full_cloud[col_id + row_id * Horizon_SCANS] = point_temp;
     }

     if (!given_offset_time)
     {
          dt_last_point = max_relative_time;

          for (int i = 0; i < full_cloud.size(); i++)
          {
               if (full_cloud[i].label >= 0)
               {
                    full_cloud[i].alpha_time = (full_cloud[i].relative_time / dt_last_point);

                    if (full_cloud[i].alpha_time > 1)
                         full_cloud[i].alpha_time = 1;
                    if (full_cloud[i].alpha_time < 0)
                         full_cloud[i].alpha_time = 0;
               }
          }
     }

     dt_offset = dt_last_point;
}

void cloudProcessing::velodyneHandler(const sensor_msgs::PointCloud2::ConstPtr &msg, double &dt_offset)
{
     pcl::PointCloud<velodyne_ros::Point> raw_cloud;
     pcl::fromROSMsg(*msg, raw_cloud);
     int size = raw_cloud.points.size();

     double dt_last_point;

     if (size == 0)
     {
          dt_offset = delta_cut_time;
          return;
     }

     if (raw_cloud.points[size - 1].time > 0)
          given_offset_time = true;
     else
          given_offset_time = false;

     if (raw_cloud.points[size - 1].ring > 0)
          given_ring = true;
     else
          given_ring = false;

     if (given_offset_time)
     {
          sort(raw_cloud.points.begin(), raw_cloud.points.end(), time_list_velodyne);

          dt_last_point = raw_cloud.points.back().time * time_unit_scale;
          delta_cut_time = dt_last_point;
     }

     double omega = 0.361 * SCAN_RATE;

     std::vector<bool> is_first;
     is_first.resize(N_SCANS);
     fill(is_first.begin(), is_first.end(), true);

     std::vector<double> yaw_first_point;
     yaw_first_point.resize(N_SCANS);
     fill(yaw_first_point.begin(), yaw_first_point.end(), 0.0);

     double max_relative_time = -1.0;
     int num_full_points = 0;

     for (int i = 0; i < size; i++)
     {
          if (std::isnan(raw_cloud.points[i].x) || std::isnan(raw_cloud.points[i].y) || std::isnan(raw_cloud.points[i].z))
               continue;

          point3D point_temp;

          point_temp.raw_point = Eigen::Vector3d(raw_cloud.points[i].x, raw_cloud.points[i].y, raw_cloud.points[i].z);
          point_temp.point = point_temp.raw_point;
          point_temp.relative_time = raw_cloud.points[i].time * time_unit_scale;

          int row_id, col_id;
          double vertical_angle, horizon_angle, yaw_angle, range;

          if (given_ring)
          {
               row_id = raw_cloud.points[i].ring;
               assert(row_id >= 0 && row_id < N_SCANS);
          }
          else
          {
               vertical_angle = atan2(point_temp.raw_point.z(), sqrt(point_temp.raw_point.x() * point_temp.raw_point.x() + point_temp.raw_point.y() * point_temp.raw_point.y())) * 180 / M_PI;

               row_id = (vertical_angle + ang_bottom) / ang_res_y;
               if (row_id < 0 || row_id >= N_SCANS)
                    continue;
          }

          yaw_angle = atan2(point_temp.raw_point.y(), point_temp.raw_point.x()) * 57.2957;

          if (is_first[row_id])
          {
               horizon_angle = atan2(point_temp.raw_point.x(), point_temp.raw_point.y()) * 180 / M_PI;

               col_id = -round((horizon_angle - 90.0) / ang_res_x) + Horizon_SCANS / 2;
               if (col_id >= Horizon_SCANS)
                    col_id -= Horizon_SCANS;

               if (col_id < 0 || col_id >= Horizon_SCANS)
                    continue;

               range = sqrt(point_temp.raw_point.x() * point_temp.raw_point.x() + point_temp.raw_point.y() * point_temp.raw_point.y() + point_temp.raw_point.z() * point_temp.raw_point.z());

               if (range < blind)
                    continue;

               point_temp.range = range;

               yaw_first_point[row_id] = yaw_angle;
               is_first[row_id] = false;

               if (given_offset_time)
               {
                    point_temp.timestamp = point_temp.relative_time / double(1000) + msg->header.stamp.toSec();
                    point_temp.alpha_time = point_temp.relative_time / dt_last_point;
               }
               else
                    point_temp.relative_time = 0.0;

               range_mat.at<float>(row_id, col_id) = range;

               num_full_points++;

               full_cloud[col_id + row_id * Horizon_SCANS] = point_temp;

               continue;
          }

          if (given_offset_time)
          {
               point_temp.timestamp = point_temp.relative_time / double(1000) + msg->header.stamp.toSec();
               point_temp.alpha_time = point_temp.relative_time / dt_last_point;
          }
          else
          {
               if (yaw_angle <= yaw_first_point[row_id])
                    point_temp.relative_time = (yaw_first_point[row_id] - yaw_angle) / omega;
               else
                    point_temp.relative_time = (yaw_first_point[row_id] - yaw_angle + 360.0) / omega;

               point_temp.timestamp = point_temp.relative_time / double(1000) + msg->header.stamp.toSec();

               if (point_temp.relative_time > max_relative_time)
                    max_relative_time = point_temp.relative_time;
          }

          horizon_angle = atan2(point_temp.raw_point.x(), point_temp.raw_point.y()) * 180 / M_PI;

          col_id = -round((horizon_angle - 90.0) / ang_res_x) + Horizon_SCANS / 2;
          if (col_id >= Horizon_SCANS)
               col_id -= Horizon_SCANS;

          if (col_id < 0 || col_id >= Horizon_SCANS)
               continue;

          range = sqrt(point_temp.raw_point.x() * point_temp.raw_point.x() + point_temp.raw_point.y() * point_temp.raw_point.y() + point_temp.raw_point.z() * point_temp.raw_point.z());

          if (range < blind)
               continue;

          point_temp.range = range;

          range_mat.at<float>(row_id, col_id) = range;

          num_full_points++;

          full_cloud[col_id + row_id * Horizon_SCANS] = point_temp;
     }

     if (!given_offset_time)
     {
          dt_last_point = max_relative_time;

          for (int i = 0; i < full_cloud.size(); i++)
          {
               if (full_cloud[i].label >= 0)
               {
                    full_cloud[i].alpha_time = (full_cloud[i].relative_time / dt_last_point);

                    if (full_cloud[i].alpha_time > 1)
                         full_cloud[i].alpha_time = 1;
                    if (full_cloud[i].alpha_time < 0)
                         full_cloud[i].alpha_time = 0;
               }
          }
     }

     dt_offset = dt_last_point;
}

void cloudProcessing::robosenseHandler(const sensor_msgs::PointCloud2::ConstPtr &msg, double &dt_offset)
{
     int ring_max = 0;

     pcl::PointCloud<robosense_ros::Point> raw_cloud;
     pcl::fromROSMsg(*msg, raw_cloud);
     int size = raw_cloud.points.size();

     double dt_last_point;

     if (size == 0)
     {
          dt_offset = delta_cut_time;
          return;
     }

     if (raw_cloud.points[size - 1].timestamp > 0)
          given_offset_time = true;
     else
          given_offset_time = false;

     if (raw_cloud.points[size - 1].ring > 0)
          given_ring = true;
     else
          given_ring = false;

     if (given_offset_time)
     {
          sort(raw_cloud.points.begin(), raw_cloud.points.end(), time_list_robosense);
          dt_last_point = (raw_cloud.points.back().timestamp - raw_cloud.points.front().timestamp) * time_unit_scale;
          delta_cut_time = dt_last_point;
     }

     double omega = 0.361 * SCAN_RATE;

     std::vector<bool> is_first;
     is_first.resize(N_SCANS);
     fill(is_first.begin(), is_first.end(), true);

     std::vector<double> yaw_first_point;
     yaw_first_point.resize(N_SCANS);
     fill(yaw_first_point.begin(), yaw_first_point.end(), 0.0);

     double max_relative_time = -1.0;
     int num_full_points = 0;

     for (int i = 0; i < size; i++)
     {
          if (std::isnan(raw_cloud.points[i].x) || std::isnan(raw_cloud.points[i].y) || std::isnan(raw_cloud.points[i].z))
               continue;

          point3D point_temp;

          point_temp.raw_point = Eigen::Vector3d(raw_cloud.points[i].x, raw_cloud.points[i].y, raw_cloud.points[i].z);
          point_temp.point = point_temp.raw_point;
          point_temp.relative_time = raw_cloud.points[i].timestamp * time_unit_scale;

          int row_id, col_id;
          double vertical_angle, horizon_angle, yaw_angle, range;

          if (given_ring)
               row_id = raw_cloud.points[i].ring;
          else
          {
               vertical_angle = atan2(point_temp.raw_point.z(), sqrt(point_temp.raw_point.x() * point_temp.raw_point.x() + point_temp.raw_point.y() * point_temp.raw_point.y())) * 180 / M_PI;

               row_id = (vertical_angle + ang_bottom) / ang_res_y;
               if (row_id > ring_max)
               {
                    ring_max = row_id;
               }

               if (row_id < 0 || row_id >= N_SCANS)
                    continue;
          }

          yaw_angle = atan2(point_temp.raw_point.y(), point_temp.raw_point.x()) * 57.2957;

          if (is_first[row_id])
          {
               horizon_angle = atan2(point_temp.raw_point.x(), point_temp.raw_point.y()) * 180 / M_PI;

               col_id = -round((horizon_angle - 90.0) / ang_res_x) + Horizon_SCANS / 2;
               if (col_id >= Horizon_SCANS)
                    col_id -= Horizon_SCANS;

               if (col_id < 0 || col_id >= Horizon_SCANS)
                    continue;

               range = sqrt(point_temp.raw_point.x() * point_temp.raw_point.x() + point_temp.raw_point.y() * point_temp.raw_point.y() + point_temp.raw_point.z() * point_temp.raw_point.z());

               if (range < blind)
                    continue;

               point_temp.range = range;

               yaw_first_point[row_id] = yaw_angle;
               is_first[row_id] = false;

               if (given_offset_time)
               {
                    point_temp.timestamp = point_temp.relative_time / double(1000) + msg->header.stamp.toSec();
                    point_temp.alpha_time = point_temp.relative_time / dt_last_point;
               }
               else
                    point_temp.relative_time = 0.0;

               range_mat.at<float>(row_id, col_id) = range;

               num_full_points++;

               full_cloud[col_id + row_id * Horizon_SCANS] = point_temp;

               continue;
          }

          if (given_offset_time)
          {
               point_temp.timestamp = point_temp.relative_time / double(1000) + msg->header.stamp.toSec();
               point_temp.alpha_time = point_temp.relative_time / dt_last_point;
          }
          else
          {
               if (yaw_angle <= yaw_first_point[row_id])
                    point_temp.relative_time = (yaw_first_point[row_id] - yaw_angle) / omega;
               else
                    point_temp.relative_time = (yaw_first_point[row_id] - yaw_angle + 360.0) / omega;

               point_temp.timestamp = point_temp.relative_time / double(1000) + msg->header.stamp.toSec();

               if (point_temp.relative_time > max_relative_time)
                    max_relative_time = point_temp.relative_time;
          }

          horizon_angle = atan2(point_temp.raw_point.x(), point_temp.raw_point.y()) * 180 / M_PI;

          col_id = -round((horizon_angle - 90.0) / ang_res_x) + Horizon_SCANS / 2;
          if (col_id >= Horizon_SCANS)
               col_id -= Horizon_SCANS;

          if (col_id < 0 || col_id >= Horizon_SCANS)
               continue;

          range = sqrt(point_temp.raw_point.x() * point_temp.raw_point.x() + point_temp.raw_point.y() * point_temp.raw_point.y() + point_temp.raw_point.z() * point_temp.raw_point.z());

          if (range < blind)
               continue;

          point_temp.range = range;

          range_mat.at<float>(row_id, col_id) = range;

          full_cloud[col_id + row_id * Horizon_SCANS] = point_temp;
     }

     if (!given_offset_time)
     {
          dt_last_point = max_relative_time;

          for (int i = 0; i < full_cloud.size(); i++)
          {
               if (full_cloud[i].label >= 0)
               {
                    full_cloud[i].alpha_time = (full_cloud[i].relative_time / dt_last_point);

                    if (full_cloud[i].alpha_time > 1)
                         full_cloud[i].alpha_time = 1;
                    if (full_cloud[i].alpha_time < 0)
                         full_cloud[i].alpha_time = 0;
               }
          }
     }

     dt_offset = dt_last_point;

     std::cout << "ring_max = " << ring_max << std::endl;
}

/**
 * @brief 地面移除函数，通过检测相邻线束间的仰角来判断地面点。
 *
 * @details
 * 该函数逐行（scan）遍历点云中的点，通过比较相邻线束的高度差和水平距离来计算两点的仰角。
 * 如果仰角与传感器安装角（sensor_mount_angle）的差值小于等于1度，则将这两个点标记为地面点。
 * 最后，将标记为地面点的坐标在 label_mat 中置为 -1，表示无需参与后续的聚类或障碍物检测。
 *
 * 变量说明：
 * - Horizon_SCANS：一帧点云的水平分辨率（如 1800、2000 等），即水平方向能量条数。
 * - ground_scan_id：地面扫描线的数量，一般只考虑激光雷达下方若干根线束数据作为地面检测对象。
 * - full_cloud：存储所有点云数据的容器，其中每个点包含其三维坐标（raw_point）和标签（label）。
 * - ground_mat：用于存储地面检测结果的矩阵（OpenCV的 Mat），其中值为 1 表示地面点，-1 表示无效点或不确定点。
 * - label_mat：最终的标记矩阵，如果某位置在 ground_mat 中被标记为 1 或 -1，则在 label_mat 中置为 -1。
 * - sensor_mount_angle：传感器安装角度，用于计算相邻线束之间的仰角阈值。
 *
 * 算法流程：
 * 1. 遍历所有扫描线（从第0线到 ground_scan_id - 1），并比较该线与下一线（i 与 i+1 线）在相同水平角度 j 的两个点。
 * 2. 如果两个点均有效（label != -1），则计算它们在三维空间的高度差和水平距离，进而得到两点的仰角 angle。
 * 3. 如果 (angle - sensor_mount_angle) 的绝对值小于等于 1，则将这两个点分别标记为地面点 (ground_mat = 1)。
 * 4. 处理完成后，再根据 ground_mat 中的标记，更新 label_mat，将地面点或无效点在 label_mat 中置为 -1。
 */
void cloudProcessing::groundRemoval()
{
     size_t lower_id, upper_id;
     float diff_x, diff_y, diff_z, angle;

     // 1. 遍历扫描线束，对相邻线束间的点进行地面检测
     for (size_t j = 0; j < Horizon_SCANS; j++)
     {
          for (size_t i = 0; i < ground_scan_id; i++)
          {
               // lower_id 与 upper_id 分别表示相邻线束在相同水平角度 j 上的两个点索引
               lower_id = j + i * Horizon_SCANS;
               upper_id = j + (i + 1) * Horizon_SCANS;

               // 如果其中任意一个点为无效点（label == -1），则无法计算仰角
               if (full_cloud[lower_id].label == -1 || full_cloud[upper_id].label == -1)
               {
                    ground_mat.at<int8_t>(i, j) = -1; // 标记为无效
                    continue;
               }

               // 2. 计算相邻点的水平距离与高度差，并求取仰角
               diff_x = full_cloud[upper_id].raw_point.x() - full_cloud[lower_id].raw_point.x();
               diff_y = full_cloud[upper_id].raw_point.y() - full_cloud[lower_id].raw_point.y();
               diff_z = full_cloud[upper_id].raw_point.z() - full_cloud[lower_id].raw_point.z();

               angle = atan2(diff_z, sqrt(diff_x * diff_x + diff_y * diff_y)) * 180 / M_PI;

               // 3. 如果仰角与传感器安装角接近(±1度)，则将两点标记为地面点
               if (std::fabs(angle - sensor_mount_angle) <= 1)
               {
                    if (ground_mat.at<int8_t>(i, j) != 1)
                         ground_mat.at<int8_t>(i, j) = 1;

                    if (ground_mat.at<int8_t>(i + 1, j) != 1)
                         ground_mat.at<int8_t>(i + 1, j) = 1;
               }
          }
     }

     // 4. 更新 label_mat，将地面点（ground_mat=1）或无效点（ground_mat=-1）在 label_mat 中标记为 -1
     for (size_t i = 0; i < N_SCANS; i++)
     {
          for (size_t j = 0; j < Horizon_SCANS; j++)
          {
               if (ground_mat.at<int8_t>(i, j) == 1 || ground_mat.at<int8_t>(i, j) == -1)
               {
                    label_mat.at<int>(i, j) = -1;
               }
          }
     }
}

void cloudProcessing::cloudSegmentation(std::vector<point3D> &v_cloud_out)
{
     int num_points = 0;

     for (size_t i = 0; i < N_SCANS; ++i)
     {
          for (size_t j = 0; j < Horizon_SCANS; j++)
          {
               if (label_mat.at<int>(i, j) == 999999)
                    continue;

               if (num_points % point_filter_num != 0)
               {
                    num_points++;
                    continue;
               }

               if (ground_mat.at<int8_t>(i, j) == 1)
               {
                    full_cloud[j + i * Horizon_SCANS].label = 0;
                    v_cloud_out.push_back(full_cloud[j + i * Horizon_SCANS]);
                    assert(!std::isnan(full_cloud[j + i * Horizon_SCANS].raw_point.x()));
                    assert(!std::isnan(full_cloud[j + i * Horizon_SCANS].raw_point.y()));
                    assert(!std::isnan(full_cloud[j + i * Horizon_SCANS].raw_point.z()));
               }
               else if (full_cloud[j + i * Horizon_SCANS].label >= 0)
               {
                    assert(ground_mat.at<int8_t>(i, j) != 1);
                    full_cloud[j + i * Horizon_SCANS].label = 1;
                    v_cloud_out.push_back(full_cloud[j + i * Horizon_SCANS]);
                    assert(!std::isnan(full_cloud[j + i * Horizon_SCANS].raw_point.x()));
                    assert(!std::isnan(full_cloud[j + i * Horizon_SCANS].raw_point.y()));
                    assert(!std::isnan(full_cloud[j + i * Horizon_SCANS].raw_point.z()));
               }

               num_points++;
          }
     }
}

void cloudProcessing::labelComponents(int row, int col)
{
     float d1, d2, alpha, angle;
     int from_id_x, from_id_y, this_id_x, this_id_y;

     bool line_count_flag[N_SCANS] = {false};

     queue_id_x[0] = row;
     queue_id_y[0] = col;

     int queue_size = 1;
     int queue_start_id = 0;
     int queue_end_id = 1;

     all_pushed_id_x[0] = row;
     all_pushed_id_y[0] = col;

     int all_pushed_id_size = 1;

     while (queue_size > 0)
     {
          from_id_x = queue_id_x[queue_start_id];
          from_id_y = queue_id_y[queue_start_id];

          queue_size--;
          queue_start_id++;

          label_mat.at<int>(from_id_x, from_id_y) = label_count;

          for (auto iter = neighbor_iterator.begin(); iter != neighbor_iterator.end(); ++iter)
          {
               this_id_x = from_id_x + (*iter).first;
               this_id_y = from_id_y + (*iter).second;

               if (this_id_x < 0 || this_id_x >= N_SCANS)
                    continue;

               if (this_id_y < 0)
                    this_id_y = Horizon_SCANS - 1;
               if (this_id_y >= Horizon_SCANS)
                    this_id_y = 0;

               if (label_mat.at<int>(this_id_x, this_id_y) != 0)
                    continue;

               d1 = std::max(range_mat.at<float>(from_id_x, from_id_y),
                             range_mat.at<float>(this_id_x, this_id_y));

               d2 = std::min(range_mat.at<float>(from_id_x, from_id_y),
                             range_mat.at<float>(this_id_x, this_id_y));

               if ((*iter).first == 0)
                    alpha = segment_alpha_x;
               else
                    alpha = segment_alpha_y;

               angle = atan2(d2 * sin(alpha), (d1 - d2 * cos(alpha)));

               if (angle > segment_theta)
               {
                    queue_id_x[queue_end_id] = this_id_x;
                    queue_id_y[queue_end_id] = this_id_y;
                    queue_size++;
                    queue_end_id++;

                    label_mat.at<int>(this_id_x, this_id_y) = label_count;
                    line_count_flag[this_id_x] = true;

                    all_pushed_id_x[all_pushed_id_size] = this_id_x;
                    all_pushed_id_y[all_pushed_id_size] = this_id_y;
                    all_pushed_id_size++;
               }
          }
     }

     bool feasible_segment = false;

     if (all_pushed_id_size >= 2)
          feasible_segment = true;

     if (feasible_segment == true)
     {
          label_count++;
     }
     else
     {
          for (size_t i = 0; i < all_pushed_id_size; i++)
               label_mat.at<int>(all_pushed_id_x[i], all_pushed_id_y[i]) = 999999;
     }
}