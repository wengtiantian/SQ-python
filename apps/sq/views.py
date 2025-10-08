"""
用户发表评分时：
（1）接口xx：输入是评论内容；输出是该评论所属SERVQUAL维度、情感得分
更新商家情况接口：
（2）接口xx：输入是所评价商家名，输出是该商家的有形性TangiSc、可靠性ReliSc、响应性ResponSc、保证性AssuranceSc、移情性EmpathySc与总得分TotalSc六个字段
与更新帕累托图、词云图\柱形图、雷达图。
"""
import json
from collections import defaultdict

import jieba
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
from common.resp import respSuccessJson
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends
from common import deps
from .curd.curd_sq import curd_sq
from .models import sqModel
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib as mpl

# --- 修改 Matplotlib 字体配置 ---
mpl.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'Heiti SC', 'SimHei', 'Arial Unicode MS']
# 解决负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False
# ---------------------------------

router = APIRouter()


class mySentDict(object):
    def __init__(self, sent_dict_path):
        with open(sent_dict_path, "r", encoding="utf-8") as f:
            self.sent_dict = json.load(f)
        self.words = set(self.sent_dict.keys())
        # print(self.words)
    def __getitem__(self, key):
        return self.sent_dict[key]
    def analyse_sent(self, review, avg=True):
        words = jieba.lcut(review)
        # 输入的评论中仅保留是存在与PMI_sent_dict文件中的词语。将这些词语存储在变量word中
        words = (set(words) & set(self.sent_dict))
        if avg:
            # 计算 words 中所有词语的情感值总和
            return sum(self.sent_dict[word] for word in words) / len(words) if len(words) > 0 else 0
        else:
            # 如果 avg 为 False，返回评论中每个词语的情感值列表
            return [self.sent_dict[word] for word in words]


def PC(all_comments,servicer_id):
    all_comments=all_comments.replace(" ", "")
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体

    # 开始绘制PC图
    words = jieba.lcut(all_comments)
    print(words)
    # 计算词频
    word_freq = pd.Series(words).value_counts(normalize=True)
    # 选择前50个高频词语
    top_words = word_freq.head(20)
    # 创建画布和坐标轴。设置图的宽高比
    fig, ax1 = plt.subplots(figsize=(14, 6))
    # 绘制柱形图
    bars = ax1.bar(top_words.index, top_words.values, color='tab:blue')

    # PC中的曲线
    ax2 = ax1.twinx()
    # 计算累积百分比
    cumulative_percentage = top_words.cumsum() / top_words.sum() * 100  # 计算累积百分比
    print(cumulative_percentage)

    # 转换为NumPy数组
    x_values = top_words.index.to_numpy()
    y_values = cumulative_percentage.to_numpy()

    ax2.plot(x_values, y_values, color='tab:orange', linewidth=3)  # 绘制累积百分比折线图

    ax1.set_title(f'帕累托图', fontsize=16, fontweight='bold')
    plt.tight_layout()
    # 显示图形
    # plt.savefig(f"../../media/Pareto_chart/{username}.jpg")
    # 主函数的相对路径
    plt.savefig(f"./media/Pareto_chart/{servicer_id}.jpg")

def Radar(servicer_id,TangiSc,ReliSc, ResponSc, AssuranceSc, EmpathySc,TotalSc):
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
    labels = np.array(['有形性', '可靠性', '响应性', '保证性', '移情性','总体'])  # 六个维度
    data = np.array([TangiSc,ReliSc, ResponSc, AssuranceSc, EmpathySc,TotalSc])  # 每个维度的值
    # data2 = np.array([10, 15, 8, 12, 9])  # 第二组数据：每个维度的评论个数
    # data2 = (data2 / data2.sum())
    # 将数据闭合，以便绘制雷达图
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)  # 计算每个维度的角度
    data = np.concatenate((data, [data[0]]))  # 闭合数据
    # data2 = np.concatenate((data2, [data2[0]]))  # 闭合数据

    angles = np.concatenate((angles, [angles[0]]))  # 闭合角度

    # 创建雷达图
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))  # 创建极坐标图

    # 绘制雷达图
    ax.plot(angles, data, 'o-', linewidth=2, label='数据')  # 绘制数据线
    ax.fill(angles, data, alpha=0.25)  # 填充颜色

    # 绘制第二组数据（评论个数）
    # ax.plot(angles, data2, 'o-', linewidth=2, label='评论个数')
    # ax.fill(angles, data2, alpha=0.25)

    # 设置角度刻度标签
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)  # 使用 angles[:-1] 避免闭合点

    # 设置雷达图的标题
    ax.set_title('SERVQUAL五个评价维度上的服务质量值', fontsize=16, fontweight='bold', pad=20)

    # 设置雷达图的刻度范围
    # ax.set_ylim(min(data)-0.1, max(data)+0.005)  # 动态设置刻度范围
    ax.set_ylim(min(data)-0.5, max(data)+0.2)  # 动态设置刻度范围

    # 显示图例
    # ax.legend(bbox_to_anchor=(1.1, 1.1))

    # 显示图形
    plt.tight_layout()
    plt.savefig(f"./media/Radar/{servicer_id}.jpg")

def Bars(servicer_id, sqdim_groups):
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题（额外添加，避免绘图异常）

    # 1. 定义满意度区间
    bins = [-float("inf"), -0.5, -0.2, 0.2, 0.5, float("inf")]
    labels = ["非常不满意", "不满意", "一般", "基本满意", "非常满意"]

    # 2. 统计每个实际存在的维度的满意度分布
    sample_counts = pd.DataFrame(columns=labels)
    for sqdim, group in sqdim_groups.items():
        sen_scores = [item.get("SenScore") for item in group if item.get("SenScore") is not None]
        sen_bins = pd.cut(sen_scores, bins=bins, labels=labels)
        bin_counts = sen_bins.value_counts().reindex(labels, fill_value=0)
        sample_counts.loc[sqdim] = bin_counts

    print(sample_counts)

    # 3. 绘图配置
    colors = ['#dd6ab0', '#7c8ebe', '#4fbb98', 'blue', 'green']
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.15

    # 生成 x 轴基础位置：长度 = 实际存在的维度数量
    x = np.arange(len(sample_counts.index))
    num_satisfaction_levels = len(sample_counts.columns)  # 满意度等级数量（固定为 5）

    # 绘制每组柱状图
    for i, (colname, col) in enumerate(sample_counts.items()):
        bars = ax.bar(x + i * bar_width, col, width=bar_width, label=colname, color=colors[i], edgecolor='black')
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=12)

    # 4. 动态生成 x 轴标签（仅显示实际存在的维度）
    # 定义维度编号到名称的映射（用于动态匹配）
    sqdim_name_map = {
        0: '有形性',
        1: '可靠性',
        2: '响应性',
        3: '保证性',
        4: '移情性'
    }
    # 动态获取标签：根据 sample_counts.index（实际维度编号）匹配名称
    dynamic_labels = [sqdim_name_map.get(idx, f'维度{idx}') for idx in sample_counts.index]
    # 计算刻度中心位置（确保每个维度的柱子居中对齐）
    tick_centers = x + bar_width * (num_satisfaction_levels - 1) / 2

    # 5. 设置 x 轴刻度
    plt.xticks(
        ticks=tick_centers,  # 刻度位置：数量 = 实际维度数
        labels=dynamic_labels,  # 刻度标签：数量 = 实际维度数
        rotation=0,
        fontsize=12
    )

    # 其他配置
    plt.legend(fontsize=12, title='满意度')
    plt.xlabel('SERVQUAL维度', fontsize=12)
    plt.ylabel('评论个数', fontsize=12)
    plt.title('SERVQUAL 维度与满意度分布', fontsize=14)
    plt.tight_layout()  # 自动调整布局，避免标签截断
    plt.savefig(f"./media/Bars/{servicer_id}.jpg")  # 保存图片
    plt.close()  # 关闭画布

def build_wordcloud(servicer_id,all_user_input):
    # 将列表转换为字符串
    text = " ".join(all_user_input)

    # 使用 jieba 进行分词
    words = " ".join(jieba.lcut(text))

    # 加载背景图（Musk 的头像）
    mask = np.array(Image.open(
        r"./alice_color.png"))  # 替换为您的图片路径
    # mask = np.array(Image.open(r"../../../词云图3.jpg"))
    # 设置词云参数
    wordcloud = WordCloud(
        font_path='simhei.ttf',  # 设置字体（支持中文）
        mask=mask,  # 设置背景图
        background_color='white',  # 背景颜色
        max_words=200,  # 最大显示词数
        max_font_size=100,  # 最大字体大小
        contour_width=3,  # 轮廓宽度
        contour_color='steelblue',  # 轮廓颜色
        prefer_horizontal=0.9,  # 水平显示比例
        relative_scaling=0.5,  # 词频与字体大小的关系
        random_state=42  # 随机种子
    )
    # 生成词云
    wordcloud.generate(words)

    # 使用背景图的颜色生成词云颜色
    image_colors = ImageColorGenerator(mask)
    wordcloud.recolor(color_func=image_colors)

    # 显示词云图
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # 隐藏坐标轴
    plt.show()
    wordcloud.to_file(f"./media/WordCloud/{servicer_id}.jpg")

def read_words_to_set(file_path):
    word_set = set()  # 用于存储 word 的集合
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 按冒号分割，提取 word
            word = line.split(":")[0].strip()  # 去除前后空白字符
            word_set.add(word)  # 添加到集合中
    return word_set

# 接口1
# 测试链接
# http://localhost:9898/api/v1/sq/ReviewProcess?current_review=售后推卸责任！ 反映问题后，商家一直拖延不处理！
@router.get("/ReviewProcess", summary="置于评论提交页面,处理用户新提交的评论,获取其所属的服务质量维度和情感极性得分")
async def getReviewDimSent(*,
                           # db: Session = Depends(deps.get_db),
                           current_review:str):
    dim1 = [
        "现代化", "整洁", "舒适", "齐全", "先进","着装", "形象", "得体", "面貌", "精神", "美观", "干净", "有序", "温馨", "宽敞", "装备", "车", "收割机",
        "农机", "电机", "员工", "师傅", "场地", "设施", "物件", "谈吐", "举止", "外表", "态度"
    ]
    # 可靠性
    dim2 = [
        "兑现", "履行", "可信", "稳定", "预期", "追溯", "透明", "口碑",
        "按时", "可靠", "记录", "发票", "账单", "合同", "便捷", "合理", "完善", "文档", "公告", "信息", "采购", "质量","说明"
    ]
    # 响应性
    dim3 = [
        "响应", "有效", "反馈", "处理", "解决", "跟进", "准时", "快速", "及时", "迅速", "即时", "高效",
        "需求", "应对", "服务", "雇佣", "维护", "检查", "播种", "肥料", "喷洒", "灌溉", "害虫", "时间", "联系","跟踪进度","紧急"
    ]
    # 保证性
    dim4 = [
        "熟练", "经验", "信赖", "安全感", "客服", "售后", "保障", "保证","标准", "合适", "规范", "信赖", "放心", "礼貌",
        "保修","专业", "资质", "突发情况", "监督"
    ]
    # 移情性
    dim5 = [
        "定制", "专属", "满足", "灵活", "贴心", "温暖", "细心", "周到", "人性化","倾听", "换位思考", "体谅", "细节", "耐心", "友好", "互动", "反馈", "信任",
        "需求", "关怀", "期盼", "个性化", "电话", "退款", "投诉", "愉快", "感激", "推荐", "沟通", "体贴", "积极","解释","宣传", "潜在", "理解", "沟通"
    ]
    dims = [dim1, dim2, dim3, dim4, dim5]
    with open(r"./apps/sq/all_dim.txt", 'r', encoding="utf-8") as file:
        lines = file.readlines()
    # 创建二维列表
    data_list = []
    for line in lines:
        line = line.strip().split()
        data_list.append([line[0], line[1], float(line[2])])
    # 即vocab
    word_set = read_words_to_set(r"./apps/sq/my_dim_dict.txt")

    words = jieba.lcut(current_review)
    words = (set(words) & word_set)
    score = []
    for dim in dims:
        weidu_sum_PMI = 0
        count = 0
        for i in data_list:
            if i[1] in words and i[0] in dim:
                weidu_sum_PMI += i[2]
                count += 1
        for j in words:
            if j in dim:
                weidu_sum_PMI += 1
                count += 1
        if count == 0:
            count = 1
        weidu_PMI = weidu_sum_PMI / float(count)
        # score存储的是评论与各个维度的PMI分
        score.append(weidu_PMI)
    # 检查 score 列表中的所有元素是否相等，相同则默认选择最后一个维度
    if all(x == score[0] for x in score):
        max_index = 4
    else:
        # 找到列表中的最大值
        max_index = score.index(max(score)) + 1
    sent_dict = mySentDict(r"./apps/sq/PMI_sent_dict2.json")
    senti_value=sent_dict.analyse_sent(current_review)
    print(max_index)
    print(senti_value)
    # 由java后台进行统一存储
    #TO DO 存储max_index
    #TO DO 存储value
    return respSuccessJson({"SQdim": max_index, "SenScore": senti_value})


# 接口2
# http://localhost:9898/api/v1/sq/SQbusiness?servicer_id=2
# 图片引用路径http://localhost:9898/media/Pareto_chart/admin.jpg
@router.get("/SQbusiness", summary="获取服务提供商在SERVQUAL各个维度的服务质量值与总体均值")
async def getSentimentInfo(*,db: Session = Depends(deps.get_db),
        servicer_id: int
):
    # [在评价表中]取出该商家在数据库中的所有评论
    filters = []
    if servicer_id:
        filters.append(sqModel.servicer_id == servicer_id)
    data = curd_sq.get_total(db, filters=filters)
    data_len=len(data)
    print(f"一个获取出了{data_len}条数据")

    # 按 SQdim 分组
    sqdim_groups = defaultdict(list)
    for item in data:
        sqdim = item.get("SQdim")
        if sqdim is not None:  # 过滤掉 SQdim 为 null 的数据
            sqdim_groups[sqdim].append(item)

    # 初始化变量
    TangiSc = None  # 维度 0
    ReliSc = None   # 维度 1
    ResponSc = None # 维度 2
    AssuranceSc = None  # 维度 3
    EmpathySc = None    # 维度 4


    # 计算每个 SQdim 组的 SenScore 均值
    for sqdim, group in sqdim_groups.items():
        # 提取 SenScore
        sen_scores = [item.get("SenScore") for item in group if item.get("SenScore") is not None]
        # 计算均值
        mean_sen_score = np.mean(sen_scores) if sen_scores else None

        # 根据 SQdim 存储到对应的变量
        if sqdim == 0:
            TangiSc = mean_sen_score
        elif sqdim == 1:
            ReliSc = mean_sen_score
        elif sqdim == 2:
            ResponSc = mean_sen_score
        elif sqdim == 3:
            AssuranceSc = mean_sen_score
        elif sqdim == 4:
            EmpathySc = mean_sen_score

    # 计算 TotalSc（前五者的均值）
    valid_scores = [score for score in [TangiSc, ReliSc, ResponSc, AssuranceSc, EmpathySc] if score is not None]
    TotalSc = np.mean(valid_scores) if valid_scores else None

    # 如果存在数据,把它们统一存进列表all_user_input中
    if data:
        all_user_input=[]
        for i in data:
            all_user_input.append(i["content"])
    # ['你好', 'arm', '什么叫做标准化', '标准化文件起草时是否有专用的编辑软件？', '基础性国家标准体系是怎么样的？', '起草标准化文件的步骤是怎么样的？原则是什么', '标准化文件的文件要素的分类和构成是怎么样的？', '初步起草一份《赣南脐橙生产技术规程》的标准化稿件。', '生成一份《地理标志产品 邻水脐橙生产技术规程》', '《晚熟脐橙标准化生产技术规程》中说了什么', '啦啦啦啦啦']
    print(all_user_input)
    all_comments = ' '.join(all_user_input)
    PC(all_comments,servicer_id)
    build_wordcloud(servicer_id,all_user_input)
    Bars(servicer_id,sqdim_groups)
    Radar(servicer_id,TangiSc,ReliSc, ResponSc, AssuranceSc, EmpathySc,TotalSc)

    return respSuccessJson({
            "TangiSc": TangiSc,
            "ReliSc": ReliSc,
            "ResponSc": ResponSc,
            "AssuranceSc": AssuranceSc,
            "EmpathySc": EmpathySc,
            "TotalSc": TotalSc,
            "len":data_len,
            "data":data
            })

