#ifndef FISK_MATH_HPP
#define FISK_MATH_HPP

#include "FiskData.hpp"
#include <random>
#include <algorithm>

class FiskMath {
public:
    static FiskData run_mice_chain(const FiskData& source,int maxit){
        FiskData current = source;
        initial_impute(current);

        std::random_device rd;
        std::mt19937 gen(rd());

        for(int iter=0;iter<maxit;iter++){
            for(int j=0;j<current.cols;j++){
                if(!has_nas(source,j)) continue;
                if(current.factor_maps[j].size()>1) impute_logistic(current,j,gen);
                else impute_pmm(current,j,gen);
            }
        }
        return current;
    }

private:
    static bool has_nas(const FiskData& fd,int col){
        for(int i=0;i<fd.rows;i++) if(std::isnan(fd.matrix(i,col))) return true;
        return false;
    }

    static void initial_impute(FiskData& fd){
        for(int j=0;j<fd.cols;j++){
            double sum=0; int count=0;
            for(int i=0;i<fd.rows;i++) if(!std::isnan(fd.matrix(i,j))){sum+=fd.matrix(i,j);count++;}
            double mean = (count>0)?sum/count:0;
            for(int i=0;i<fd.rows;i++) if(std::isnan(fd.matrix(i,j))) fd.matrix(i,j)=mean;
        }
    }

    static void impute_pmm(FiskData& fd,int target_col,std::mt19937& gen){
        int n=fd.rows; int p=fd.cols;
        Eigen::MatrixXd X = fd.matrix; Eigen::VectorXd y=X.col(target_col);

        std::vector<int> obs, mis;
        for(int i=0;i<n;i++) (std::isnan(y(i))?mis:obs).push_back(i);

        if(obs.size()<5) return;

        Eigen::MatrixXd Xobs(obs.size(),p); Eigen::VectorXd yobs(obs.size());
        for(int i=0;i<obs.size();i++){ Xobs.row(i)=X.row(obs[i]); yobs(i)=y(obs[i]); }

        Eigen::VectorXd beta=(Xobs.transpose()*Xobs).ldlt().solve(Xobs.transpose()*yobs);
        Eigen::VectorXd pred_obs=Xobs*beta;

        int k=5; std::uniform_int_distribution<> rand_k(0,k-1);
        for(int idx:mis){
            double pred_mis = X.row(idx).dot(beta);
            std::vector<std::pair<double,int>> dist;
            for(int i=0;i<obs.size();i++) dist.push_back({std::abs(pred_obs(i)-pred_mis),obs[i]});
            std::sort(dist.begin(),dist.end());
            int chosen = dist[rand_k(gen)].second;
            fd.matrix(idx,target_col)=y(chosen);
        }
    }

    static void impute_logistic(FiskData& fd,int target_col,std::mt19937& gen){
        int n=fd.rows; int p=fd.cols;
        Eigen::MatrixXd X=fd.matrix; Eigen::VectorXd y=X.col(target_col);
        std::vector<int> obs, mis;
        for(int i=0;i<n;i++) (std::isnan(y(i))?mis:obs).push_back(i);

        if(obs.size()<5) return;
        Eigen::MatrixXd Xobs(obs.size(),p); Eigen::VectorXd yobs(obs.size());
        for(int i=0;i<obs.size();i++){ Xobs.row(i)=X.row(obs[i]); yobs(i)=y(obs[i]); }

        Eigen::VectorXd beta = (Xobs.transpose()*Xobs).ldlt().solve(Xobs.transpose()*yobs);
        std::uniform_real_distribution<double> dist(0.0,1.0);
        for(int idx:mis){
            double p=1.0/(1.0+exp(-X.row(idx).dot(beta)));
            fd.matrix(idx,target_col)=(dist(gen)<p)?1.0:0.0;
        }
    }
};

#endif