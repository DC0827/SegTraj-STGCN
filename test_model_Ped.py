import numpy as np
import random
import Model
from data_loader_Ped import data_generator
from utils import *
from args import args
from main_Ped import args_config_Ped
from scipy.ndimage import gaussian_filter1d



def load_trained_model(model_path, args):
    model = Model.FC_STGNN_Pedestrians_FC(
        args.indim_fea,
        args.conv_out, 
        args.lstmhidden_dim, 
        args.lstmout_dim,
        args.conv_kernel,
        args.hidden_dim,
        args.time_denpen_len, 
        args.num_windows,
        args.moving_window,
        args.stride, 
        args.decay, 
        args.pool_choice, 
        args.num_heads
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

# 判定是否需要画箭头
def is_static_trajectory(traj, x_range, y_range, ratio_thresh=0.15):
    x_span = traj[:, 0].max() - traj[:, 0].min()
    y_span = traj[:, 1].max() - traj[:, 1].min()
    return (x_span / x_range < ratio_thresh) and (y_span / y_range < ratio_thresh)

# 平滑轨迹美观展示
def smooth_traj_simple(traj, sigma=1.0):
    traj[:, 0] = gaussian_filter1d(traj[:, 0], sigma=sigma)
    traj[:, 1] = gaussian_filter1d(traj[:, 1], sigma=sigma)
    return traj



def visualize_samples(data_batch, preds, trues, metrics_list, vis_embeddings, save_path, tag, overall_metrics):
 
    def draw_group(ax, group_points, group_id, color, label=None):
        group_points = np.array(group_points)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        width  = x_range * 0.18
        height = y_range * 0.18

        for pt in group_points:
            ellipse = plt.matplotlib.patches.Ellipse(
                pt, width=width, height=height,
                angle=0, color=color, alpha=0.3,
                zorder=0.5, clip_on=False
            )
            ax.add_patch(ellipse)

        ax.scatter(group_points[:, 0], group_points[:, 1],
                s=80, color=color, edgecolors='k', linewidths=0.8,
                label=label, zorder=3)
        
   
  

    n_show = min(5, len(preds))
    fig, axes = plt.subplots(2, n_show, figsize=(5 * n_show, 10))
    title = f"{tag} Dataset Trajectory Visualization\nOverall P={overall_metrics[0]:.2f} R={overall_metrics[1]:.2f} F1={overall_metrics[2]:.2f} Acc={overall_metrics[3]:.2f}%"
    fig.suptitle(title, fontsize=18)

    colors = plt.get_cmap("tab10")
    true_color_maps = []
    pred_color_maps = []

    for idx in range(n_show):
        data = data_batch[idx].cpu().numpy()
        data = np.transpose(data, (1, 0, 2, 3))
        data_reshaped = data.reshape(data.shape[0], data.shape[1]*data.shape[2], data.shape[3])
        coords = data_reshaped[:, :, :2]

        pred_groups = preds[idx]
        true_groups = trues[idx]
        P, R, F, acc = metrics_list[idx]

        true_group_colors = {gid: colors(gid % 10) for gid in range(len(true_groups))}
        true_color_maps.append(true_group_colors)

        used_colors = set(true_group_colors.values())
        pred_group_colors = {}
        assigned_true = set()
        for pred_gid, pred_group in enumerate(pred_groups):
            overlaps = [len(set(pred_group) & set(true_group)) for true_group in true_groups]
            best_match = np.argmax(overlaps)
            if overlaps[best_match] > 0 and best_match not in assigned_true:
                pred_group_colors[pred_gid] = true_group_colors[best_match]
                assigned_true.add(best_match)
            else:
                for cid in range(10):
                    color = colors(cid)
                    if color not in used_colors:
                        pred_group_colors[pred_gid] = color
                        used_colors.add(color)
                        break
                else:
                    pred_group_colors[pred_gid] = colors(pred_gid % 10)
        pred_color_maps.append(pred_group_colors)

        x_all = coords[:, :, 0].flatten()
        y_all = coords[:, :, 1].flatten()
        x_min, x_max = x_all.min(), x_all.max()
        y_min, y_max = y_all.min(), y_all.max()
        x_pad = (x_max - x_min) * 0.1 + 1e-3
        y_pad = (y_max - y_min) * 0.1 + 1e-3

        ax_true = axes[0, idx]
        for gid, group in enumerate(true_groups):
            color = true_group_colors[gid]
            for i, pid in enumerate(group):
                traj = coords[pid]
                # 稍微平滑轨迹美观展示
                traj = smooth_traj_simple(traj, sigma=1.2)
                label = f'Group {gid}' if i == 0 else None
                ax_true.plot(traj[:-1, 0], traj[:-1, 1], linestyle='-', color=color,linewidth=2.3,  label=label)
                ax_true.scatter(traj[:-2, 0], traj[:-2, 1], color=color, s=30)
                # 轨迹太短就不画箭头了，太乱了
                if not is_static_trajectory(traj, x_max-x_min, y_max-y_min):
                    ax_true.annotate('', xy=traj[-1], xytext=traj[-2], arrowprops=dict(arrowstyle='-|>,head_width=0.5,head_length=0.8', linewidth=2, color=color))
                else:
                    ax_true.scatter(traj[-1, 0], traj[-1, 1],s=50, color=color)
        ax_true.set_title(f"Ground Truth Sample {idx}", fontsize=12)
        ax_true.set_xlim(x_min - x_pad, x_max + x_pad)
        ax_true.set_ylim(y_min - y_pad, y_max + y_pad)
        ax_true.legend(fontsize=8)

        ax_pred = axes[1, idx]
        for gid, group in enumerate(pred_groups):
            color = pred_group_colors[gid]
            for i, pid in enumerate(group):
                traj = coords[pid]
                # 稍微平滑轨迹美观展示
                traj = smooth_traj_simple(traj, sigma=1.2)
                label = f'Group {gid}' if i == 0 else None
                ax_pred.plot(traj[:-1, 0], traj[:-1, 1], linestyle='-', color=color,linewidth=2.3, label=label)
                ax_pred.scatter(traj[:-2, 0], traj[:-2, 1], color=color, s=30)
                if not is_static_trajectory(traj, x_max-x_min, y_max-y_min):
                    ax_pred.annotate('', xy=traj[-1], xytext=traj[-2], arrowprops=dict(arrowstyle='-|>,head_width=0.5,head_length=0.8', linewidth=2, color=color))
                else:
                    ax_pred.scatter(traj[-1, 0], traj[-1, 1],s=50, color=color)
        ax_pred.set_title(f"Predicted Sample {idx}\nP:{P:.2f} R:{R:.2f} F1:{F:.2f} Acc:{acc:.2f}%", fontsize=10)
        ax_pred.set_xlim(x_min - x_pad, x_max + x_pad)
        ax_pred.set_ylim(y_min - y_pad, y_max + y_pad)
        ax_pred.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path)
    plt.close()

    # ===== Embedding Visualization =====
    if vis_embeddings is not None:
        fig_emb, axes_emb = plt.subplots(2, n_show, figsize=(5 * n_show, 10))
        fig_emb.suptitle(
            f"{tag} Dataset Node Embeddings\nOverall P:{overall_metrics[0]:.2f} R:{overall_metrics[1]:.2f} F1:{overall_metrics[2]:.2f} Acc:{overall_metrics[3]:.1f}%",
            fontsize=16
        )

        if n_show == 1:
            axes_emb = np.expand_dims(axes_emb, axis=0)

        for idx in range(n_show):
            embeddings = vis_embeddings[idx]
            true_groups = trues[idx]
            pred_groups = preds[idx]
            true_group_colors = true_color_maps[idx]
            pred_group_colors = pred_color_maps[idx]

            num_nodes = sum(len(g) for g in true_groups)
            embeddings = embeddings[:num_nodes]

            perplexity = min(5, len(embeddings) - 1)
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            emb_2d = reducer.fit_transform(embeddings)

            x_all = emb_2d[:, 0]
            y_all = emb_2d[:, 1]
            x_min, x_max = x_all.min(), x_all.max()
            y_min, y_max = y_all.min(), y_all.max()
            x_pad = max((x_max - x_min) * 0.15, 1)
            y_pad = max((y_max - y_min) * 0.15, 1)

            ax_gt = axes_emb[0, idx]
            ax_gt.set_xlim(x_min - x_pad, x_max + x_pad)
            ax_gt.set_ylim(y_min - y_pad, y_max + y_pad)
            for gid, group in enumerate(true_groups):
                color = true_group_colors[gid]
                group_points = emb_2d[group]
                draw_group(ax_gt, group_points, gid, color, label=f"G{gid}")
            ax_gt.set_title(f"Ground Truth Sample {idx}", fontsize=10)
            ax_gt.legend(fontsize=7, markerscale=0.6, handlelength=1, handletextpad=0.5, borderaxespad=0.3)

            ax_pred = axes_emb[1, idx]
            ax_pred.set_xlim(x_min - x_pad, x_max + x_pad)
            ax_pred.set_ylim(y_min - y_pad, y_max + y_pad)
            for gid, group in enumerate(pred_groups):
                color = pred_group_colors.get(gid, colors(gid % 10))
                group_points = emb_2d[group]
                draw_group(ax_pred, group_points, gid, color, label=f"G{gid}")
            P, R, F, acc = metrics_list[idx]
            ax_pred.set_title(f"Predicted Sample {idx}\nP:{P:.2f} R:{R:.2f} F1:{F:.2f} A:{acc:.1f}%", fontsize=9)
            ax_pred.legend(fontsize=7, markerscale=0.6, handlelength=1, handletextpad=0.5, borderaxespad=0.3)

        fig_emb.tight_layout(rect=[0, 0, 1, 0.92])
        fig_emb.savefig(save_path.replace(".svg", "_embeddings.svg"), dpi=300)
        plt.close(fig_emb)





def test_model(model, test_loader, tag="Test"):
    model.eval()
    prediction_, label_, data_batch_,node_embeddings_ = [], [], [],[]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            label_.extend(label.cpu().numpy())
            data_batch_.extend(data.cpu())
            adjacency_matrix, _,node_embeddings = model(data)
            prediction = find_groups_method2(adjacency_matrix)
            prediction_.extend(prediction)
            node_embeddings_.extend(node_embeddings.cpu().numpy())

    total_P, total_R, total_F, total_pairwise_accuracy = 0, 0, 0, 0
    n = len(prediction_)
    selected_samples = random.sample(range(n), min(num_samples, n))
    print(f"\n===== 随机选取的{tag}样本预测结果 =====")

    metrics_list = []
    vis_preds, vis_trues, vis_data, vis_embeddings = [], [], [],[]

    for i, (y_bar, y) in enumerate(zip(prediction_, label_)):
        nodes_num = sum(len(group) for group in y_bar)
        y = tensor_to_array_groups(y, nodes_num)
        P, R, F = compute_groupMitre(y, y_bar)
        pairwise_accuracy = compute_pairwise_accuracy(y, y_bar)

        total_pairwise_accuracy += pairwise_accuracy
        total_P += P
        total_R += R
        total_F += F

        if i in selected_samples:
            print(f"样本:{i}")
            print(f"预测分组: {y_bar}")
            print(f"真实分组: {y}")
            print(f"P: {P:.4f}, R: {R:.4f}, F1: {F:.4f} , pairwise_accuracy: {pairwise_accuracy:.4f}%")
            vis_preds.append(y_bar)
            vis_trues.append(y)
            vis_data.append(data_batch_[i])
            vis_embeddings.append(node_embeddings_[i]) 
            metrics_list.append((P, R, F, pairwise_accuracy))

    avg_P = total_P / n
    avg_R = total_R / n
    avg_F = total_F / n
    avg_pairwise_accuracy = total_pairwise_accuracy / n

    if vis_data:
        visualize_samples(vis_data, vis_preds, vis_trues, metrics_list,vis_embeddings,
                          f"./visualization/{DATASET_NAME}/visual_results_{tag}.svg",tag,
                          (avg_P, avg_R, avg_F, avg_pairwise_accuracy))

    print(f"Test Results: P={avg_P:.4f}, R={avg_R:.4f}, F1={avg_F:.4f} , pairwise_accuracy={avg_pairwise_accuracy:.4f}%")
    return avg_P, avg_R, avg_F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    args = args()
    args = args_config_Ped(args)

    # 可变参数,待处理数据集的名称

    dataset_list = ["ETH", "Hotel", "students03"]
    num_samples = 5

    for DATASET_NAME in dataset_list:
        print(f"\n=== Evaluating dataset: {DATASET_NAME} ===")
        
        MODEL_PATH = f"./saved_models/best_model_{DATASET_NAME}.pth"
        
        try:
            train_loader, val_loader, test_loader = data_generator(f"./data/{DATASET_NAME}/", args=args)
            model = load_trained_model(MODEL_PATH, args)
            
            test_model(model, train_loader, tag="Train")
            test_model(model, val_loader, tag="Val")
            test_model(model, test_loader, tag="Test")
            
            print(f"模型总参数量: {count_parameters(model):,}")
        except Exception as e:
            print(f"[Error] 处理 {DATASET_NAME} 时出错：{e}")