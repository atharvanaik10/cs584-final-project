import argparse
import os
import torch                # 2个interfface的 详细用法，包括算法参数化，请参见 
import time
import copy
import sys
from plnn.network_linear_approximation import LinearizedNetwork
from plnn.anderson_linear_approximation import AndersonLinearizedNetwork
from tools.cifar_bound_comparison import load_network, make_elided_models, cifar_loaders, dump_bounds
from tools.bab_tools.model_utils import load_cifar_l1_network

from plnn.simplex_solver.solver import SimplexLP
from plnn.simplex_solver.baseline_solver import Baseline_SimplexLP
from plnn.simplex_solver import utils
from plnn.simplex_solver.baseline_gurobi_linear_approximation import Baseline_LinearizedNetwork
from plnn.simplex_solver.gurobi_linear_approximation import Simp_LinearizedNetwork
from plnn.simplex_solver.disjunctive_gurobi import DP_LinearizedNetwork

import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Compute and time a bunch of bounds.")
    parser.add_argument('--network_filename', type=str, ### 指定网络模型文件的path
                        help='Path to the network')
    parser.add_argument('--eps', type=float,            ### 代表扰动幅度（epsilon）
                        help='Epsilon - default: 0.5')
    parser.add_argument('--target_directory', type=str, ### 实验结果的保存目录
                        help='Where to store the results')
    parser.add_argument('--modulo', type=int,           ### 分割dataset的 job数量
                        help='Numbers of a job to split the dataset over.')
    parser.add_argument('--modulo_do', type=int,        ### job_id
                        help='Which job_id is this one.')
    parser.add_argument('--from_intermediate_bounds', action='store_true',  ### 
                        help="if this flag is true, intermediate bounds are computed w/ best of naive-KW")
    parser.add_argument('--nn_name', type=str, help='network architecture name')
    args = parser.parse_args()

    np.random.seed(0)

    if args.nn_name:    # 加载model
        model = load_cifar_l1_network(args.nn_name)
    else:
        model = load_network(args.network_filename)

    results_dir = args.target_directory     #### 结果储存dict
    os.makedirs(results_dir, exist_ok=True)
    elided_models = make_elided_models(model, True)   ### 换掉model的最后一个layer
        ### 换成 算出来的prediction weight，与ground truth的差值【是论文p3的objective function】
        # objective function是 prediction和ground truth的差值。 如果其>0，就是预测correct
            # 求min是因为～～【看看L8 ？？？？？？？】
    # elided_models = make_elided_models(model)

    planet_correct_sum = 0
    dp_correct_sum = 0

    total_images = 0
    gur_planet_correct_sum = 0
    gur_dp_correct_sum = 0

    basline_bigm_adam_correct_sum = 0
    basline_cut_correct_sum = 0

    ft = open(os.path.join(results_dir, "ver_acc.txt"), "a")      ### 这是写入所有最终结果的文件
                # 将指定目录results_dir与文件名ver_acc.txt 连接起来来构建完整路径，然后打开文件

    _, test_loader = cifar_loaders(1) ### 返回为数据的data loaders, 即DataLoader instance（之后会用这些loader批量处理图像）

    for idx, (X, y) in enumerate(test_loader):
        ###### 每个循环都会过一遍下面的所有方法 ######
            ### idx：处理的当前批次的index
            ### X：这表示从 CIFAR-10数据集加载的一批图像
            ### y：这是对应于 X 中每幅图像的一批labels
        #处理来自test_loader的每个图像。如果被指定的话，用modulo条件将工作分配到不同的jobs。这在任务分布在多个 processors或会话的场景中使用
        if (args.modulo is not None) and (idx % args.modulo != args.modulo_do):
            continue

        if idx>=1000:   # “只load这么多图片？？？”，可删减数量
            sys.exit()

            ### 为每张处理过的图像在结果目录中创建一个小目录，用于存储特定图像的处理结果
        target_dir = os.path.join(results_dir, f"{idx}") 
        os.makedirs(target_dir, exist_ok=True)


        ###### 进行predicting ######      # 每张图像模型都会做出预测
        out = model(X)             
            ### softmax应用于沿dim的所有切片，并将重新缩放这些切片，使元素位于[0,1]范围内,且总和为1
        pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
        print(idx, y.item(), pred[0])
        if y.item()!=pred[0]:       ### 如果预测与真实label匹配，将会继续计算perturbed image周围的 bounds
            # print("Incorrect prediction")     # y 是原输出值
            continue        # ？？？

        total_images +=1


        elided_model = elided_models[y.item()]  #？？？
        to_ignore = y.item()

        domain = torch.stack([X.squeeze(0) - args.eps,      # 所有输入值的domain，扰动值eps的范围
                              X.squeeze(0) + args.eps], dim=-1).unsqueeze(0)

        lin_approx_string = "" if not args.from_intermediate_bounds else "-fromintermediate"




        """#######################################
        ###### FOR BASELINE METHODS FIRST #####         baseline方法【可以先跑通这第一个，然后exist】
        #######################################         BASELINE 是什么？？？

        # compute intermediate bounds. Use these for every method to allow comparison on the last layer
        cuda_elided_model = copy.deepcopy(elided_model).cuda()
        intermediate_net = Baseline_SimplexLP([lay for lay in cuda_elided_model], max_batch=3000)
        cuda_domain = (X.cuda(), args.eps)  # 把damain存在cuda上，gpu
        domain = (X, args.eps) # 把damain存在cpu上

        grb_start = time.time()

        with torch.no_grad():
            intermediate_net.set_solution_optimizer("best_naive_simplex", None)
            intermediate_net.define_linear_approximation(cuda_domain, no_conv=False)
        intermediate_ubs = intermediate_net.upper_bounds
        intermediate_lbs = intermediate_net.lower_bounds




        #### baseline-lirpa Bounds ####    # 用到lirpa, 来自alpha-bata crown的一部分
        # use regular lirpa,  convex barriaer bound
        lirpa_target_file = os.path.join(target_dir, f"baseline-lirpa6{lin_approx_string}-fixed.txt")
        lirpa_l_target_file = os.path.join(target_dir, f"l_baseline-lirpa6{lin_approx_string}-fixed.txt")
        if not os.path.exists(lirpa_l_target_file):
            lirpa_params = {
                "nb_outer_iter": 6
            }
            lirpa_net = Baseline_SimplexLP(cuda_elided_model, params=lirpa_params,  # ？？？
                                 store_bounds_progress=len(intermediate_net.weights), debug=True)
            lirpa_start = time.time()
            with torch.no_grad():
                lirpa_net.optimize = lirpa_net.auto_lirpa_optimizer
                lirpa_net.logger = utils.OptimizationTrace()
                lirpa_net.build_model_using_intermediate_net(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
                lb, ub = lirpa_net.compute_lower_bound()
            lirpa_end = time.time()
            lirpa_time = lirpa_end - lirpa_start
            lirpa_lbs = lb.detach().cpu()   #算出来了lower，upper bound，存入cpu
            lirpa_ubs = ub.detach().cpu()
            dump_bounds(lirpa_target_file, lirpa_time, lirpa_ubs)
            dump_bounds(lirpa_l_target_file, lirpa_time, lirpa_lbs)

            ###########################
            ## verified accuracy    # 计算bounds后评估验证精度，即计算出的 adversarial robustness的upper bounds是否在可接受的范围内。
            correct=1
            for bn in ub.cpu()[0]:
                if bn >0:
                    correct=0
                    break
            planet_correct_sum += correct
            ###########################
            del lirpa_net




        # ########################### ？？？
        ### pgd bounds
        pgd_target_file = os.path.join(target_dir, f"pgd{lin_approx_string}-fixed.txt")
        l_pgd_target_file = os.path.join(target_dir, f"l_pgd{lin_approx_string}-fixed.txt")

        pgd_bounds = intermediate_net.advertorch_pgd_upper_bound()
        pgd_time = 0
        pgd_lbs = pgd_bounds[:9]
        pgd_ubs = pgd_bounds[9:]
        dump_bounds(pgd_target_file, pgd_time, pgd_ubs)
        dump_bounds(l_pgd_target_file, pgd_time, pgd_lbs)
        # ###########################


        # Gurobi baseline-planet-simplex Bounds    
        grb_target_file = os.path.join(target_dir, f"gurobi-baseline-planet-simplex{lin_approx_string}-fixed.txt")
        # grb_l_target_file = os.path.join(target_dir, f"l_gurobi-baseline-planet-simplex{lin_approx_string}-fixed.txt")
        if not os.path.exists(grb_target_file):
            grb_net = Baseline_LinearizedNetwork([lay.cpu() for lay in elided_model])
            grb_start = time.time()
            if not args.from_intermediate_bounds:
                grb_net.define_linear_approximation(domain, n_threads=4)
                ub = grb_net.upper_bounds[-1]
            else:
                grb_net.build_model_using_bounds(domain, ([lbs.cpu().squeeze(0) for lbs in intermediate_lbs], [ubs.cpu().squeeze(0) for ubs in intermediate_ubs]), n_threads=4)
                _ , ub = grb_net.compute_lower_bound(ub_only=True)
            grb_end = time.time()
            grb_time = grb_end - grb_start
            grb_ubs = torch.Tensor(ub).cpu()
            # grb_lbs = torch.Tensor(lb).cpu()
            dump_bounds(grb_target_file, grb_time, grb_ubs)
            # dump_bounds(grb_l_target_file, grb_time, grb_lbs)
            del grb_net

            ###########################
            ## verified accuracy
            correct=1
            for bn in ub.cpu()[0]:
                if bn >0:
                    correct=0
                    break
            gur_planet_correct_sum += correct
            ###########################











        ######################################################
        ################ DUAL BOUNDS ###################
        ######################################################

        ## baseline-bigm-adam-simplex. 
        for bigm_steps in [850]:
            bigm_adam_params = {
                "bigm_algorithm": "adam",
                "bigm": "only",
                "nb_outer_iter": bigm_steps,
                'initial_step_size': 1e-2,
                'final_step_size': 1e-4,
                'betas': (0.9, 0.999)
            }
            bigm_target_file = os.path.join(target_dir, f"baseline-bigm-adam-simplex_{bigm_steps}{lin_approx_string}.txt")
            if not os.path.exists(bigm_target_file):
                cuda_elided_model = copy.deepcopy(elided_model).cuda()
                bigm_net = Baseline_SimplexLP(cuda_elided_model, params=bigm_adam_params)
                bigm_start = time.time()
                with torch.no_grad():
                    bigm_net.optimize = bigm_net.bigm_subgradient_optimizer
                    bigm_net.logger = utils.OptimizationTrace()
                    if not args.from_intermediate_bounds:
                        bigm_net.define_linear_approximation(cuda_domain)
                        ub = bigm_net.upper_bounds[-1]
                    else:
                        bigm_net.build_model_using_intermediate_net(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
                        _, ub = bigm_net.compute_lower_bound()
                bigm_end = time.time()
                bigm_time = bigm_end - bigm_start
                bigm_ubs = ub.cpu()

                del bigm_net
                dump_bounds(bigm_target_file, bigm_time, bigm_ubs)

                ###########################
                ## verified accuracy
                correct=1
                for bn in ub.cpu()[0]:
                    if bn >0:
                        correct=0
                        break
                basline_bigm_adam_correct_sum += correct
                ###########################


        # baseline-active-set-simplex. 
        bigm_adam_params = {
            "bigm_algorithm": "adam",
            "bigm": "only",
            "nb_outer_iter": 500,
            'initial_step_size': 1e-2,
            'final_step_size': 1e-4,
            'betas': (0.9, 0.999)
        }
        bigm_target_file = os.path.join(target_dir, f"baseline-cut-simplex_{lin_approx_string}.txt")
        if not os.path.exists(bigm_target_file):
            cuda_elided_model = copy.deepcopy(elided_model).cuda()
            bigm_net = Baseline_SimplexLP(cuda_elided_model, params=bigm_adam_params)
            bigm_start = time.time()
            with torch.no_grad():
                bigm_net.optimize = bigm_net.cut_anderson_optimizer
                bigm_net.logger = utils.OptimizationTrace()
                if not args.from_intermediate_bounds:
                    bigm_net.define_linear_approximation(cuda_domain)
                    ub = bigm_net.upper_bounds[-1]
                else:
                    bigm_net.build_model_using_intermediate_net(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
                    _, ub = bigm_net.compute_lower_bound()
            bigm_end = time.time()
            bigm_time = bigm_end - bigm_start
            bigm_ubs = ub.cpu()

            del bigm_net
            dump_bounds(bigm_target_file, bigm_time, bigm_ubs)

            ###########################
            ## verified accuracy
            correct=1
            for bn in ub.cpu()[0]:
                if bn >0:
                    correct=0
                    break
            basline_cut_correct_sum += correct
            ###########################

        del intermediate_net
        """








        # 注释掉后，然后去跑，能不能跑通。可能不用grobi。
        # 根据这两文件的代码延伸去看，不要乱看其他代码，比如说script里面的其他文件。


        #######################################
        ### SIMPLEX VERIFY METHODS ###
        #######################################

        ### Computing Intermediate bounds
        cuda_elided_model = copy.deepcopy(elided_model).cuda()  ### 复制elided_model并将其移至 GPU
        intermediate_net = SimplexLP([lay for lay in cuda_elided_model], max_batch=3000) 
                ### 初始化一个中intermediate_network, 是SimplexLP的一个instance，并且设置了最大batch大小
        cuda_domain = (X.cuda(), args.eps)  ### 为GPU和CPU版本设置domain
        domain = (X, args.eps)

        grb_start = time.time() ### 记录当前时间，作为验证过程执行时间的基准。

        with torch.no_grad():
            intermediate_net.set_solution_optimizer('best_naive_simplex', None)
            intermediate_net.define_linear_approximation(cuda_domain, no_conv=False,
                                                         override_numerical_errors=True)
        intermediate_ubs = intermediate_net.upper_bounds
        intermediate_lbs = intermediate_net.lower_bounds


        # # ## auto-lirpa-dp Bounds
        lirpa_target_file = os.path.join(target_dir, f"auto-lirpa-dp-3{lin_approx_string}-fixed.txt")
        lirpa_l_target_file = os.path.join(target_dir, f"l_auto-lirpa-dp-3{lin_approx_string}-fixed.txt")
        if not os.path.exists(lirpa_l_target_file):
            lirpa_params = {
                "nb_outer_iter": 3,
            }
            lirpa_net = SimplexLP(cuda_elided_model, params=lirpa_params,
                             store_bounds_progress=len(intermediate_net.weights), debug=True, dp=True)
            lirpa_start = time.time()
            with torch.no_grad():
                lirpa_net.optimize = lirpa_net.auto_lirpa_optimizer
                lirpa_net.logger = utils.OptimizationTrace()
                lirpa_net.build_model_using_intermediate_net(cuda_domain, (intermediate_lbs, intermediate_ubs), intermediate_net)
                lb, ub = lirpa_net.compute_lower_bound()
            lirpa_end = time.time()
            lirpa_time = lirpa_end - lirpa_start
            lirpa_lbs = lb.detach().cpu()
            lirpa_ubs = ub.detach().cpu()
            dump_bounds(lirpa_target_file, lirpa_time, lirpa_ubs)
            dump_bounds(lirpa_l_target_file, lirpa_time, lirpa_lbs)

            # print(ub)
            ###########################
            ## verified accuracy
            correct=1
            for bn in ub.cpu()[0]:
                if bn >0:
                    correct=0
                    break
            dp_correct_sum += correct
            ###########################
            
            del lirpa_net



        ## Gurobi dp-simplex Bounds
        grb_target_file = os.path.join(target_dir, f"gurobi-dp-simplex{lin_approx_string}-fixed.txt")
        # grb_l_target_file = os.path.join(target_dir, f"l_gurobi-dp-simplex{lin_approx_string}-fixed.txt")
        if not os.path.exists(grb_target_file):
            grb_net = DP_LinearizedNetwork([lay for lay in elided_model], intermediate_net.weights)
            grb_start = time.time()
            if not args.from_intermediate_bounds:
                grb_net.define_linear_approximation(domain, n_threads=4)
                ub = grb_net.upper_bounds[-1]
            else:
                grb_net.build_model_using_bounds(domain, ([lbs.cpu().squeeze(0) for lbs in intermediate_lbs], [ubs.cpu().squeeze(0) for ubs in intermediate_ubs]), intermediate_net.init_cut_coeffs, n_threads=4)
                _, ub = grb_net.compute_lower_bound(ub_only=True)
            grb_end = time.time()
            grb_time = grb_end - grb_start
            grb_ubs = torch.Tensor(ub).cpu()
            # grb_lbs = torch.Tensor(lb).cpu()
            dump_bounds(grb_target_file, grb_time, grb_ubs)
            # dump_bounds(grb_l_target_file, grb_time, grb_lbs)
            del grb_net

            ###########################
            ## verified accuracy
            correct=1
            for bn in ub.cpu()[0]:
                if bn >0:
                    correct=0
                    break
            gur_dp_correct_sum += correct
            ###########################
        del intermediate_net










        # 打印所有的结果

        print('Nominal acc: ', total_images/float(idx+1), 'Planet, simplex_verify acc: ', planet_correct_sum/float(idx+1), dp_correct_sum/float(idx+1))

        print('Gurobi Planet, dp acc: ', gur_planet_correct_sum/float(idx+1), gur_dp_correct_sum/float(idx+1))

        print('Bigm-adam, cut acc: ', basline_bigm_adam_correct_sum/float(idx+1), basline_cut_correct_sum/float(idx+1))

        



        ########    写入文件

        ########
        ft.write(str(planet_correct_sum))
        ft.write(",")
        ft.write(str(dp_correct_sum))
        ft.write(",")
        ft.write(str(gur_planet_correct_sum))
        ft.write(",")
        ft.write(str(gur_dp_correct_sum))
        ft.write(",")
        ft.write(str(basline_bigm_adam_correct_sum))
        ft.write(",")
        ft.write(str(basline_cut_correct_sum))
        ft.write(",")
        ft.write(str(total_images))
        ft.write(",")
        ft.write(str(idx))
        ft.write("\n")


if __name__ == '__main__':
    main()
