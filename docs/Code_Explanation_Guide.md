脑瘤 MRI 分类项目讲解文档
对应 notebook：Brain_Tumor_Detection_Colab_educational_report.ipynb
用途：这份文档是给汇报和答辩准备的，不是论文版。重点是让你能用大白话讲清楚：每一步为什么存在、核心代码在做什么、领导随机问到时怎么回答。
# 一、项目一句话说明
这个项目的目标是：用公开脑部 MRI 图片训练一个 ResNet18 二分类模型，判断一张 MRI 图像属于有脑瘤（YES）还是无脑瘤（NO）。
# 二、整体流程
## 0. Environment Setup：环境和数据集准备
这一节在干什么：安装缺少的小工具包，下载 Kaggle 数据集，并设置工作目录。
核心代码怎么读：
LIGHTWEIGHT_PACKAGES = [("kagglehub", "kagglehub"), ...]
这里列出项目需要但 Colab 可能没有的小包。每一组前面是安装名，后面是 import 时用的名字。
if importlib.util.find_spec(module_name) is None:
先检查这个包能不能被 Python 找到；找不到才安装，避免重复安装。
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])
让当前 Python 环境执行 pip install。-q 是 quiet，意思是少输出一点安装日志。
DATASET_ROOT = Path(kagglehub.dataset_download(DATASET_SLUG))
从 Kaggle 下载数据集，并把下载位置变成 Path 路径对象。
DATA_DIR = DATASET_ROOT / "brain_tumor_dataset"
定位真正装图片的文件夹，里面应该有 yes 和 no 两个子文件夹。
大白话理解：
- pip install 就像给 Python 装插件；没有这个包，后面 import 就会报 not defined 或 module not found。
- Path 是专门处理路径的工具，DATASET_ROOT / 'brain_tumor_dataset' 表示在数据集根目录下面继续找 brain_tumor_dataset 文件夹。
- 工作目录是当前代码默认读写文件的地方，后面生成 TRAIN、VAL、TEST 文件夹都会在这里。
领导可能问：
问：为什么要自动安装包？
答：因为 Colab 环境不一定自带 kagglehub、imutils、tqdm。自动检查并安装可以让 notebook 更容易从头运行。
问：DATASET_SLUG 是什么？
答：它是 Kaggle 数据集的名字，相当于数据集地址的简写。
## 1. Imports and Random Seed：导入库和固定随机种子
这一节在干什么：把后面会用到的工具库导进来，并固定随机性。
核心代码怎么读：
import random, time, copy
random 用来打乱文件；time 用来计时；copy 用来保存模型最佳状态。
import numpy as np
NumPy 用来处理图片数组和标签数组。
import torch / torchvision
PyTorch 和 torchvision 用来搭建、训练 ResNet 模型。
RANDOM_SEED = 47
设置一个固定数字，让随机打乱、训练初始化尽量可复现。
torch.manual_seed(RANDOM_SEED)
固定 PyTorch 里的随机数。
大白话理解：
- 随机种子不是模型参数，它只是让随机过程尽量每次一致。
- 比如洗牌，如果不固定种子，每次洗出来顺序都不同；固定种子后，每次洗牌顺序更稳定。
领导可能问：
问：为什么要固定随机种子？
答：为了让实验结果更稳定，别人重跑时更接近我的结果。
问：PyTorch 是干什么的？
答：它是深度学习框架，负责张量计算、模型搭建、反向传播和参数更新。
## 2. Dataset Check：检查数据集
这一节在干什么：确认数据目录里真的有 yes 和 no 两类图片。
核心代码怎么读：
assert (DATA_DIR / "yes").exists()
如果 yes 文件夹不存在，程序立刻报错提醒。
files = [p for p in (DATA_DIR / class_name).iterdir() if p.is_file() and not p.name.startswith(".")]
遍历类别文件夹，只保留普通文件，并过滤掉隐藏文件。
print(f"{class_name}: {len(files)} images")
打印每个类别有多少张图片。
大白话理解：
- 这一步不是训练，是防止数据路径错了。
- yes 表示有脑瘤，no 表示无脑瘤；后面会把这两个文件夹名转成数字标签。
领导可能问：
问：怎么看出是正常文件？
答：代码用 p.is_file() 判断它是文件，不是文件夹；not p.name.startswith('.') 过滤掉系统隐藏文件。
问：为什么要先检查？
答：因为如果路径错了，后面增强、划分、训练都会跟着错。
## 3-4. Data Augmentation：数据增强
这一节在干什么：把原始 MRI 图片做轻微变化，生成更多训练样本。
核心代码怎么读：
def data_augment(img_dir, n_samples, save_to_dir):
定义一个函数：从 img_dir 读取图片，每张生成 n_samples 张增强图，保存到 save_to_dir。
transforms.RandomRotation(15)
随机旋转图片，最多约 15 度。
transforms.RandomAffine(... translate=(0.05, 0.05), shear=5, scale=(0.95, 1.05))
随机平移、轻微错切、轻微缩放，让图片有合理变化。
transforms.RandomHorizontalFlip(p=0.5)
有 50% 概率水平翻转。
for aug_idx in range(n_samples):
对每张原图重复生成指定数量的增强图片。
N_AUG_PER_IMAGE = 6
每张原图生成 6 张增强图，再加 1 张原图副本。
大白话理解：
- 增强不是凭空造医学事实，而是让模型看到同一类图片的轻微角度变化。
- n_samples 是函数调用时传进去的数字，这里是 6。
- enumerate 会同时给图片一个编号和图片路径，但编号主要方便以后扩展或调试。
领导可能问：
问：为什么要数据增强？
答：原始数据量较小，增强可以增加训练样本的多样性，减少模型死记硬背。
问：增强会不会改变标签？
答：不会。轻微旋转、平移、翻转后，有脑瘤还是有脑瘤，无脑瘤还是无脑瘤。
## 5. Train / Validation / Test Split：划分数据集
这一节在干什么：把数据拆成训练集、验证集、测试集。
核心代码怎么读：
TEST_RATIO = 0.15 / VAL_RATIO = 0.15
测试集占 15%，验证集占 15%，剩下约 70% 用来训练。
random.shuffle(files)
把同一类别的图片顺序打乱，避免按文件名顺序切分造成偏差。
test_files = files[:n_test]
前一部分作为测试集。
val_files = files[n_test:n_test + n_val]
中间一部分作为验证集。
train_files = files[n_test + n_val:]
剩下的作为训练集。
shutil.copy2(src, dst)
把图片复制到 TRAIN、VAL、TEST 对应文件夹。
大白话理解：
- 训练集用来学习；验证集用来看训练过程中效果好不好；测试集留到最后做最终评估。
- 不能只用训练集看结果，因为模型可能只是记住训练图片。
领导可能问：
问：为什么要分三份？
答：训练集负责学习，验证集负责调参和观察，测试集负责最后检验泛化能力。
问：为什么要 shuffle？
答：防止文件本来的排列顺序影响划分结果。
## 6. Load Images into Python Arrays：加载图片
这一节在干什么：把文件夹里的图片读进 Python，变成模型前期处理可以用的数组。
核心代码怎么读：
IMG_SIZE = (224, 224)
统一图片大小为 224x224，这是 ResNet 常用输入尺寸。
class_dirs = sorted([p for p in dir_path.iterdir() if p.is_dir()])
找出当前目录下面的类别文件夹，比如 NO 和 YES。
for label_id, class_dir in enumerate(class_dirs):
给每个类别文件夹分配数字标签，比如 NO=0，YES=1。
img = Image.open(img_path).convert("RGB").resize(img_size)
打开图片，转成 RGB 三通道，并缩放到统一大小。
X.append(np.asarray(img, dtype=np.uint8))
把图片变成 NumPy 数组，放进 X。
y.append(label_id)
把这张图片对应的数字标签放进 y。
大白话理解：
- X 是图片数据，y 是答案标签。
- 文件夹里的图片变成 Python 数组 X；文件夹名字 YES/NO 变成数字标签 y。
- uint8 表示像素值是 0 到 255 的整数，符合普通图片的存储方式。
领导可能问：
问：为什么要把图片变成数组？
答：深度学习模型不能直接理解 jpg 文件，它只能处理数字矩阵。
问：为什么标签要变成数字？
答：模型输出也是数字类别，所以 YES/NO 要转成 0/1 这种形式。
## 7. Class Distribution Bar Chart：类别柱状图
这一节在干什么：用柱状图展示训练集、验证集、测试集中 YES 和 NO 的数量。
核心代码怎么读：
id_to_label = {label_id: class_name for class_name, label_id in labels.items()}
把 labels 字典反过来，方便从数字找到类别名。
np.sum(y_split == class_id)
统计某个数据集中某个类别有多少张。
x = np.arange(len(split_names))
生成柱状图 x 轴位置，比如 Train、Val、Test 三个位置。
ax.bar(...)
画柱状图。
大白话理解：
- 这张图可以给领导看数据分布是否离谱。
- 如果某个类别特别少，模型可能偏向数量多的类别。
领导可能问：
问：np.arange 是什么？
答：它会生成一串数字位置，比如 np.arange(3) 得到 0、1、2，用来放三组柱子。
问：这张图有什么汇报价值？
答：它说明数据划分后各类别数量是否基本合理。
## 8-9. Simple Preprocessing and Save：简单预处理并保存
这一节在干什么：保证所有图片都是 RGB、uint8、224x224，然后保存成新的训练文件夹。
核心代码怎么读：
def crop_imgs(set_name, add_pixels_value=0, img_size=IMG_SIZE):
这里保留函数名 crop_imgs，但实际做的是简单 resize，不再做复杂轮廓裁剪。
img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
把像素限制在 0 到 255，并转成图片常用格式。
cv2.resize(img_uint8, img_size)
把图片统一成 224x224。
save_new_images(X_train_crop, y_train, labels, "TRAIN_CROP")
把处理后的训练图片保存到 TRAIN_CROP 文件夹。
大白话理解：
- 这一版已经降低难度，不做复杂的脑部轮廓裁剪。
- 保留 X_train_crop 这些变量名，是为了让后面的训练流程不用大改。
领导可能问：
问：这一部分重要吗？
答：重要，但现在已经简化了。核心作用是统一输入格式，保证后面模型能稳定读取。
问：为什么还叫 crop_imgs？
答：为了兼容原 notebook 的变量和 TODO 结构，实际上这版主要是 resize。
## 10. Define ResNet Input Transforms：ResNet 输入 transform（IMAGENET_MEAN 在这里）
这一节在干什么：定义 PyTorch 读取图片时要做的处理：resize、增强、转 Tensor、标准化。
核心代码怎么读：
IMAGENET_MEAN = [0.485, 0.456, 0.406]
ImageNet 数据集上常用的 RGB 均值。
IMAGENET_STD = [0.229, 0.224, 0.225]
ImageNet 数据集上常用的 RGB 标准差。
transforms.ToTensor()
把 PIL 图片转成 PyTorch Tensor，并把像素从 0-255 缩放到 0-1。
transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
按 ImageNet 的统计方式标准化，因为我们用的是 ImageNet 预训练模型。
train_transform vs eval_transform
训练时有随机增强；验证和测试时不加随机增强，保证评估稳定。
大白话理解：
- ImageNet 是大规模自然图片数据集；ResNet18 先在 ImageNet 上学过通用图像特征。
- 我们不是先 ImageNet 再 ResNet，而是用“已经在 ImageNet 上训练好的 ResNet”。
- 迁移学习就是把大模型学到的基础能力拿过来，再改成自己的任务。
领导可能问：
问：为什么要 Normalize？
答：预训练 ResNet 当初就是这样处理输入的。我们用同样的标准，模型更容易正常工作。
问：为什么训练和验证 transform 不一样？
答：训练要增强多样性，验证和测试要稳定评估，所以不加随机变化。
## 11. Dataset and DataLoader：数据集和批量读取（ImageFolder/DataLoader 在这里）
这一节在干什么：让 PyTorch 按批次把图片和标签送进模型。
核心代码怎么读：
ImageFolder("TRAIN_CROP", transform=train_transform)
按文件夹名自动识别类别，并对图片应用 transform。
DataLoader(..., batch_size=BATCH_SIZE, shuffle=True)
每次取一小批图片训练，并打乱训练顺序。
dataset_sizes = {name: len(image_datasets[name]) for name in image_datasets}
记录每个数据集有多少张图片，用来计算平均 loss 和 accuracy。
class_names = image_datasets["train"].classes
读取类别名，比如 NO、YES。
大白话理解：
- DataLoader 像一个发牌器，每次给模型发一批图片，而不是一次性把所有图片都塞进去。
- batch_size=8 表示一次训练 8 张图片。
领导可能问：
问：为什么不一次训练全部图片？
答：一次全部训练占内存大，也不利于梯度更新。分批训练更常见。
问：shuffle 为什么只在训练集开？
答：训练需要打乱顺序，验证和测试通常不需要。
## 12. Training Function：训练函数
这一节在干什么：定义模型训练的循环：前向预测、计算误差、反向传播、更新参数、记录结果。
核心代码怎么读：
for epoch in range(num_epochs):
训练多轮，每一轮模型看一遍训练数据。
for phase in ["train", "val"]:
每一轮分训练阶段和验证阶段。
model.train() / model.eval()
训练模式会更新模型；评估模式不更新模型。
outputs = model(inputs)
把图片输入模型，得到预测结果。
loss = criterion(outputs, labels_batch)
计算预测和真实标签之间的差距。
loss.backward()
反向传播，计算每个参数应该怎么改。
optimizer.step()
根据梯度真正更新参数。
best_model_wts = copy.deepcopy(model.state_dict())
保存验证集表现最好的一次模型参数。
大白话理解：
- 模型训练就是：先猜答案，看错多少，再根据错误调整自己。
- 梯度可以理解成“参数应该往哪个方向改，才能让错误变小”。
- 训练两种阶段不是训练两遍：train 是学习，val 是考试，不改参数。
领导可能问：
问：什么是 loss？
答：loss 是模型错得有多严重的数字，越小越好。
问：什么是梯度？
答：梯度告诉模型参数往哪个方向调，能让 loss 降低。
问：为什么要保存 best model？
答：最后一轮不一定最好，所以保存验证集准确率最高的版本。
## 13. Build ResNet18 Model：建立 ResNet18 模型
这一节在干什么：加载预训练 ResNet18，并把最后分类层改成适合 YES/NO 的二分类。
核心代码怎么读：
models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
加载官方预训练 ResNet18。
for param in model_ft.parameters(): param.requires_grad = False
冻结大部分参数，不让它们训练。
model_ft.fc = nn.Linear(num_ftrs, len(class_names))
替换最后一层，让输出类别数变成 2。
optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=1e-3)
只训练最后一层分类器，学习率是 0.001。
大白话理解：
- ResNet18 是模型结构；ImageNet 是它以前学习过的大数据集。
- 冻结前面层，是因为前面已经会识别边缘、纹理、形状等基础特征。
- 我们主要训练最后一层，让它从通用识别变成脑瘤 YES/NO 分类。
领导可能问：
问：为什么不用从零训练？
答：数据量小，从零训练容易效果差。迁移学习可以利用已有特征。
问：为什么只训练最后一层？
答：这样更简单、训练更快，也降低过拟合风险。
## 14. Train the Model and Plot Curves：训练并画曲线
这一节在干什么：开始训练模型，并用曲线展示 loss 和 accuracy 的变化。
核心代码怎么读：
NUM_EPOCHS = 3
训练 3 轮，适合教学演示；正式实验可以增加轮数。
model_ft, history = train_model(...)
调用训练函数，返回训练好的模型和历史记录。
axes[0].plot(history["train_loss"])
画训练 loss 曲线。
axes[1].plot(history["val_acc"])
画验证准确率曲线。
大白话理解：
- loss 下降通常说明模型在学习。
- accuracy 上升说明预测正确的比例变高。
- 如果 train 很好但 val 很差，可能是过拟合。
领导可能问：
问：训练曲线怎么看？
答：主要看 loss 是否下降、accuracy 是否上升，以及 train 和 val 差距是否过大。
问：为什么只训练 3 轮？
答：这是教学和汇报版本，先保证流程清楚；后续可以增加 epoch 做更充分训练。
## 15. Visualize Prediction Results：展示预测结果
这一节在干什么：随机展示几张验证集图片和模型预测，方便汇报时直观看效果。
核心代码怎么读：
def imshow(inp, title=None):
把标准化后的 Tensor 还原成图片显示。
model.eval()
预测展示时切到评估模式，不更新参数。
_, preds = torch.max(outputs, 1)
取模型输出中分数最高的类别作为预测结果。
visualize_model(model_ft, num_images=6)
展示 6 张验证图片和对应预测类别。
大白话理解：
- 这一步不是训练，只是把模型预测结果可视化给人看。
- 它适合放进 PPT，因为比单纯数字更直观。
领导可能问：
问：预测结果图有什么作用？
答：它能让听众直观看到模型把哪些 MRI 判成 YES 或 NO。
问：为什么用 validation 图片展示？
答：验证集不是直接用来更新参数的，比训练集展示更有参考意义。
# 三、汇报时可以这样讲
我这次做的是一个脑瘤 MRI 图像二分类项目。整体上，我先从 Kaggle 下载公开数据集，检查 yes 和 no 两类图片是否存在；然后通过数据增强增加样本多样性；接着把数据划分成训练集、验证集和测试集；之后把图片统一成 ResNet18 能读取的 224x224 RGB 格式；最后使用 ImageNet 预训练的 ResNet18 做迁移学习，只训练最后的分类层，实现 YES/NO 二分类。
我在汇报版 notebook 中保留了三个结果展示：第一是类别柱状图，用来说明数据划分是否相对平衡；第二是训练曲线，用来观察 loss 和 accuracy 的变化；第三是预测结果图，用来直观看模型对验证集图片的判断。
# 四、最容易被问到的问题速记
- 为什么用 ResNet18？：它是经典 CNN 模型，结构成熟，预训练权重容易调用，适合教学和小项目迁移学习。
- ImageNet 和 ResNet 是什么关系？：ImageNet 是大数据集，ResNet18 是模型；我们用的是在 ImageNet 上预训练过的 ResNet18。
- 为什么要数据增强？：原始 MRI 图片数量有限，增强能让模型看到更多合理变化，减少过拟合。
- 为什么要分训练集、验证集、测试集？：训练集学习，验证集观察训练效果，测试集最后评估泛化能力。
- 为什么图片要 resize 到 224x224？：ResNet 常用输入尺寸是 224x224，统一尺寸后才能批量送入模型。
- 为什么要 Normalize？：预训练模型当初使用 ImageNet 的均值和标准差，所以新任务输入也保持同样标准。
- 什么是梯度？：梯度告诉模型参数应该往哪个方向调整，才能让错误 loss 变小。
- 为什么只训练最后一层？：前面层保留预训练的通用特征，最后一层改成适合脑瘤 YES/NO 的分类器。

# 表格内容

## 表格 1
阶段 | 做什么 | 你汇报时可以怎么说
数据准备 | 下载数据、检查 yes/no 文件夹 | 先保证数据集结构正确，否则后面的训练都没有意义。
数据增强 | 对图片做旋转、平移、翻转 | 原始数据少，所以用轻微变化扩充训练数据，提高泛化能力。
数据划分 | 分成 TRAIN / VAL / TEST | 训练集学习，验证集调试，测试集最后评估。
加载与预处理 | 把图片读成数组，统一成 224x224 | 模型只能吃固定大小的数字矩阵，所以要标准化输入。
ResNet 训练 | 使用 ImageNet 预训练 ResNet18 | 用迁移学习减少从零训练的成本，更适合小数据集。
结果展示 | 柱状图、训练曲线、预测结果 | 用图说明数据是否平衡、模型是否在学习、预测效果是否直观。