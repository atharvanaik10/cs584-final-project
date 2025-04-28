import argparse
import os
import torch                # 2个interfface的 详细用法，包括算法参数化，请参见 
import time
import copy
import sys
#from plnn.network_linear_approximation import LinearizedNetwork
#from plnn.anderson_linear_approximation import AndersonLinearizedNetwork
from tools.cifar_bound_comparison import load_network, make_elided_models, cifar_loaders, dump_bounds
###from tools.bab_tools.model_utils import load_cifar_l1_network

from plnn.simplex_solver.solver import SimplexLP
#from plnn.simplex_solver.baseline_solver import Baseline_SimplexLP
from plnn.simplex_solver import utils
#from plnn.simplex_solver.baseline_gurobi_linear_approximation import Baseline_LinearizedNetwork
#from plnn.simplex_solver.gurobi_linear_approximation import Simp_LinearizedNetwork
#from plnn.simplex_solver.disjunctive_gurobi import DP_LinearizedNetwork

import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Compute and time a bunch of bounds.")
    parser.add_argument('--network_filename', type=str, ### 指定网络模型文件的path【是./data/cifar_sgd_8px.pth，基于实验的名字，需要修改为relu_model.pth】
                        help='Path to the network')
    parser.add_argument('--eps', type=float,            ### 代表扰动幅度（epsilon）【对应eps {epsilon}】
                        help='Epsilon - default: 0.5')
    parser.add_argument('--target_directory', type=str, ### 实验结果的保存目录【基于实验的名字】
                        help='Where to store the results')
    parser.add_argument('--modulo', type=int,           ### 分割dataset的 job数量【没参数】
                        help='Numbers of a job to split the dataset over.')
    parser.add_argument('--modulo_do', type=int,        ### job_id【没参数】
                        help='Which job_id is this one.')
    parser.add_argument('--from_intermediate_bounds', action='store_true',  ### 【无需参数】
                        help="if this flag is true, intermediate bounds are computed w/ best of naive-KW")
    parser.add_argument('--nn_name', type=str, help='network architecture name')    ### 【没参数】
    args = parser.parse_args()

    np.random.seed(0)


    model = load_network(args.network_filename)
    #### 已经改为corwn的relu_model.pth了 ####
    #######################################


    results_dir = args.target_directory     #### 结果储存dict
    os.makedirs(results_dir, exist_ok=True)

    ### 换掉model的最后一个layer
    elided_models = make_elided_models(model, True)
        ### 换成 算出来的prediction weight，与ground truth的差值【是论文p3的objective function】
        # objective function是 prediction和ground truth的差值。 如果其>0，就是预测correct
    #elided_models = make_elided_models(model)

    dp_correct_sum = 0
    total_images = 0

    ### 这是写入所有最终结果的文件
    ft = open(os.path.join(results_dir, "ver_acc.txt"), "a")      
                # 将指定目录results_dir与文件名ver_acc.txt 连接起来来构建完整路径，然后打开文件

    _, test_loader = cifar_loaders(1) ### 返回为数据的data loaders, 即DataLoader instance（之后会用这些loader批量处理图像）
    ### test_loader是分batch的数据，每个batch由data和正确label组成



    for idx, (X, y) in enumerate(test_loader):
        ###### 每个循环都会过一遍下面的所有方法 ######
            ### idx：处理的当前批次的index
            ### X：这表示从 CIFAR-10数据集加载的一批图像
            ### y：这是对应于 X 中每幅图像的一批labels
        #处理来自test_loader的每个图像。如果被指定的话，用modulo条件将工作分配到不同的jobs。这在任务分布在多个 processors或会话的场景中使用
        if (args.modulo is not None) and (idx % args.modulo != args.modulo_do):
            continue

        if idx>=1000:   # “只load这么多图片”，可删减数量
            sys.exit()

        ### 为每张处理过的图像在结果目录中创建一个小目录，用于存储特定图像的处理结果
        target_dir = os.path.join(results_dir, f"{idx}") 
        os.makedirs(target_dir, exist_ok=True)


        ###########################
        ####### 进行predicting ######      
        ###########################
        ### 每张图像模型都会做出预测

        ### 是这个神经网络的output
        out = model(X) 
        
        ### softmax应用于沿dim的所有切片，并将重新缩放这些切片，使元素位于[0,1]范围内,且总和为1
        pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
        print(idx, y.item(), pred[0])

        ### 如果预测与真实label匹配，将会继续计算perturbed image周围的 bounds
        if y.item()!=pred[0]:
            # print("Incorrect prediction")     # y 是原输出值
            continue 
        ### 如果预测出差了，就可以之间抛弃掉，直接continue去下一个图片了。
        ### 只有预测对了，才会去后面verify。才会对它做一些扰动，去verify。  因为如果预测错了就没必要去verify

        total_images +=1
        elided_model = elided_models[y.item()]
            #### 需要用到 ####
        to_ignore = y.item()
        domain = torch.stack([X.squeeze(0) - args.eps,      # 所有输入值的domain，扰动值eps的范围
                              X.squeeze(0) + args.eps], dim=-1).unsqueeze(0)

        lin_approx_string = "" if not args.from_intermediate_bounds else "-fromintermediate"





        #######################################
        ### SIMPLEX VERIFY METHODS ###
        #######################################

        ### Computing Intermediate bounds
            # 开始算中间层的bound

        ### 复制elided_model并将其移至 GPU
        cuda_elided_model = copy.deepcopy(elided_model).cuda()  

        ### 初始化一个中intermediate_network, 是SimplexLP的一个instance，并且设置了最大batch大小
        #   对每个layer算一遍
        intermediate_net = SimplexLP([lay for lay in cuda_elided_model], max_batch=3000) 

        ### 为GPU和CPU版本设置domain   
        cuda_domain = (X.cuda(), args.eps)      ### 这个X来自test_loader    ？？？？？？？？？？？？
        #domain = (X, args.eps)
        #grb_start = time.time()    ### 记录当前时间，作为验证过程执行时间的基准。

        with torch.no_grad():   ### 不计算
            intermediate_net.set_solution_optimizer('best_naive_simplex', None)

            ### 存储中间层bounds和weights, 只处理linearlayer
            intermediate_net.define_linear_approximation(cuda_domain, no_conv=False,
                                                         override_numerical_errors=True)
        ### 得到linear中间层的上下bounds，是一系列bounds数据（在define_linear_approximation内计算完成的）
        intermediate_ubs = intermediate_net.upper_bounds
        intermediate_lbs = intermediate_net.lower_bounds
        """
        先求中间层的上下限, 在之后的build_model_using_intermediate_net函数要用到

        这部分是做了论文的 problem 5的内容（不是, problem 5也是属于论文后面的算法的一部分）

        1. 从上一层开始算这一层。每层都用planet算法来算
        2. 从第0层开始, 之间算出来n-1层了。
        """
        #l1 norm可能是只是中间layer的时候用的，因为后面的没有用到x值
        """
        cuda_domain在下面虽然作为参数, 但没有用到

        与l1 norm相关，是输入值与l1 norm相关（可能和define_linear_approximation相关）

        auto_LiRPA是一个用于自动推导和计算神经网络边界的库，它使用基于线性松弛的扰动分析 (LiRPA)
            - 就是根据输入的函数找着两根线
        """



        ### auto-lirpa-dp Bounds ###
            # 开始算最后一层的bound

        ### 设置path，调用lirpa的libirary
        lirpa_target_file = os.path.join(target_dir, f"auto-lirpa-dp-3{lin_approx_string}-fixed.txt")
        lirpa_l_target_file = os.path.join(target_dir, f"l_auto-lirpa-dp-3{lin_approx_string}-fixed.txt")

        ### 如果没有算过，即没有文件，就开始算
        if not os.path.exists(lirpa_l_target_file):
            ### iteration#，也许是伪代码的t_max
            lirpa_params = {    
                "nb_outer_iter": 3,
            }

            ### 调用class的constructor，完成了伪代码的SIMPLEX_VERIFY()函数的初始数据设置
            lirpa_net = SimplexLP(cuda_elided_model, params=lirpa_params,
                             store_bounds_progress=len(intermediate_net.weights), debug=True, dp=True)

            lirpa_start = time.time()
            with torch.no_grad():
                ### 这里只是设置optimizer，赋值
                lirpa_net.optimize = lirpa_net.auto_lirpa_optimizer
                lirpa_net.logger = utils.OptimizationTrace()

                # 为何这里构建新model的时候需要用到之前计算的中间层bound，而不是在auto_lirpa_optimizer迭代时 计算出的中间层bound
                        # 也许是因为在优化weight的时候的参数不能用
                lirpa_net.build_model_using_intermediate_net(cuda_domain,
                                                             (intermediate_lbs, intermediate_ubs), 
                                                             intermediate_net)

                ### 调用优化器(如auto-lirpa）对最后一层neuron的 bound 进行精细优化(tightening)  
                    # 最后一层只是一系列输入值，需要通过bound来选出最优的几个。 
                    # 这个函数里使用了上面的lirpa_net.optimize，即最优输出bounds
                lb, ub = lirpa_net.compute_lower_bound()
                    ### 这个调用完成了SIMPLEX_VERIFY()函数的所有计算
                    # 它进行多次迭代，对weight进行优化；每次迭代都循环所有layer来得到输出bounds
                    #  最后它不会储存优化过的weigth系数，只会返回在最优weigtht系数的场合 计算出的model的最优输出bounds
                
            lirpa_end = time.time()
            lirpa_time = lirpa_end - lirpa_start

            ### 最终得到的上下限
            lirpa_lbs = lb.detach().cpu()   ### pytorch，从tensor obj得到真正的上下限
            lirpa_ubs = ub.detach().cpu()

            dump_bounds(lirpa_target_file, lirpa_time, lirpa_ubs)   ### 写入file
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

   

        # 打印加速的比较结果
        #print('Nominal acc: ', total_images/float(idx+1), 
        #      'Planet, simplex_verify acc: ', planet_correct_sum/float(idx+1), dp_correct_sum/float(idx+1))
        #print('Gurobi Planet, dp acc: ', gur_planet_correct_sum/float(idx+1), gur_dp_correct_sum/float(idx+1))
        #print('Bigm-adam, cut acc: ', basline_bigm_adam_correct_sum/float(idx+1), basline_cut_correct_sum/float(idx+1))

        ########    写入文件
        ########
        #ft.write(str(planet_correct_sum))
        #ft.write(",")
        ft.write(str(dp_correct_sum))
        ft.write(",")
        #ft.write(str(gur_planet_correct_sum))
        #ft.write(",")
        #ft.write(str(gur_dp_correct_sum))
        #ft.write(",")
        #ft.write(str(basline_bigm_adam_correct_sum))
        #ft.write(",")
        #ft.write(str(basline_cut_correct_sum))
        #ft.write(",")
        ft.write(str(total_images))
        ft.write(",")
        ft.write(str(idx))
        ft.write("\n")


if __name__ == '__main__':
    main()
