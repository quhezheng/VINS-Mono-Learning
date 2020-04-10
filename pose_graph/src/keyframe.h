#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "parameters.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"

#define MIN_LOOP_NUM 25

using namespace Eigen;
using namespace std;
using namespace DVision;

/**
* @class BriefExtractor
* @Description 通过Brief模板文件，对图像的关键点计算Brief描述子
*/
class BriefExtractor
{
public:
  virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;
  BriefExtractor(const std::string &pattern_file);

  DVision::BRIEF m_brief;
};

/**
* @class KeyFrame
* @Description 构建关键帧，通过BRIEF描述子匹配关键帧和回环候选帧
*/
class KeyFrame
{
public:
	KeyFrame(const std_msgs::Header &_header, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
			 vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_normal, 
			 vector<double> &_point_id, int _sequence);
	KeyFrame(const std_msgs::Header &_header, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
			 cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
			 vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors);
	bool findConnection(KeyFrame* old_kf);
	void computeWindowBRIEFPoint();
	void computeBRIEFPoint();
	//void extractBrief();
	int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);
	bool searchInAera(const BRIEF::bitset window_descriptor,
	                  const std::vector<BRIEF::bitset> &descriptors_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old_norm,
	                  cv::Point2f &best_match,
	                  cv::Point2f &best_match_norm);
	void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
						  std::vector<cv::Point2f> &matched_2d_old_norm,
                          std::vector<uchar> &status,
                          const std::vector<BRIEF::bitset> &descriptors_old,
                          const std::vector<cv::KeyPoint> &keypoints_old,
                          const std::vector<cv::KeyPoint> &keypoints_old_norm);
	void FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                const std::vector<cv::Point2f> &matched_2d_old_norm,
                                vector<uchar> &status);
	void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
	               const std::vector<cv::Point3f> &matched_3d,
	               std::vector<uchar> &status,
	               Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);
	void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info);

	Eigen::Vector3d getLoopRelativeT();	//当前帧相对匹配的loop帧坐标系下的位移
	double getLoopRelativeYaw();	// 当前帧相对匹配的loop帧坐标系下的yaw
	Eigen::Quaterniond getLoopRelativeQ();	// 当前帧相对匹配的loop帧坐标系下的旋转姿态


	std_msgs::Header header;
	//double time_stamp; 
	int index;
	int local_index;	// 仅仅在PoseGraph::optimize4DoF()回环图优化中索引局部待优化变量数组用到
	Eigen::Vector3d vio_T_w_i; // 未回环优化的VOI坐标系下的坐标(VoiPose)
	Eigen::Matrix3d vio_R_w_i; // 未回环优化的VOI坐标系下的旋转(VoiPose)
	Eigen::Vector3d T_w_i;  // 回环优化过的世界坐标系下的坐标(Pose)
	Eigen::Matrix3d R_w_i;  // 回环优化过的世界坐标系下的旋转(Pose)
	Eigen::Vector3d origin_vio_T;		
	Eigen::Matrix3d origin_vio_R;
	cv::Mat image;
	cv::Mat thumbnail;
	vector<cv::Point3f> point_3d; //经过前端光流剔除后的关键点3d坐标
	vector<cv::Point2f> point_2d_uv; //经过前端光流剔除后的关键点uv坐标
	vector<cv::Point2f> point_2d_norm; //经过前端光流剔除后的关键点归一化坐标
	vector<double> point_id; //经过前端光流剔除后的关键点全局累计id
	vector<cv::KeyPoint> keypoints;			//整个图像的关键点，进行两帧匹配时为了尽量全面匹配而使用
	vector<cv::KeyPoint> keypoints_norm;
	vector<cv::KeyPoint> window_keypoints;	//经过前端光流剔除后的关键点
	vector<BRIEF::bitset> brief_descriptors; //keypoints关键点BRIEF描述子
	vector<BRIEF::bitset> window_brief_descriptors;//经过前端光流剔除后的关键点BRIEF描述子
	bool has_fast_point;
	int sequence;	//所属图像关键帧序列的id,除非图像时间上中间出现断开的问题，所有关键帧该序列id都一样

	bool has_loop;	//是否检测匹配到回环
	int loop_index;	//回环检测到的目标帧的index
	Eigen::Matrix<double, 8, 1 > loop_info; // [x,y,z, qw, qx, qy, qz, yaw]， cur帧相对old帧坐标系下的位移+姿态+yaw == T_old_cur. 单独保存yaw是为了4DOF回环图优化中使用
};

