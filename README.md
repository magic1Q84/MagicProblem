python 添加包
----------------
2020年10月7日 由 samstar
这是第一篇用来记录学习过程的博客，大约是年纪大了记忆力不够了2333
 
第一次用mac上的pycharm，上来就发现python是2.7，似乎python2已经停用有了，果断换成3，然后报错
不能安装该软件，因为当前无法从软件更新服务器获得
百度之后发现似乎是mac os系统的问题，去developer.apple.com下载了一个Command Line Tools就好了，具体原因没搞懂2333，反正弄好了
接下来添加opencv包
右下角很小的一个python那个按钮，这里可以添加包，点那个interpreter settings
 
打开之后点加号就完了，不出意料又出了问题，下载不能
百度说要改源，点Manage Repositories可以更改源（改源大法好）

清华: https://pypi.tuna.tsinghua.edu.cn/simple

豆瓣: http://pypi.douban.com/simple/

阿里: http://mirrors.aliyun.com/pypi/simple/
别忘了下载的时候在选项加入信任选项，看起来这玩意显然是信任阿里云
--trusted-host mirrors.aliyun.com
有时候下载包会缺失文件，mac中python包的安装位置如下
 





遇到了一大堆玄学问题之后不知道怎么就好了23333



python读取excel
--------------
代码：
import pandas as pd

path = 'test.xlsx'
data = pd.read_excel(path,header=None,names=['P','N'])
#header:指定作为列名的行，None表示没有列名  names：指定列名列表

print(data.head())
输出：
   P   N
0  1 NaN
1  2 NaN
2  3 NaN





python adta.describe函数
2020年10月13日 由 samstar
describe() 函数可以查看数据的基本情况，包括：count 非空值数、mean 平均值、std 标准差、max 最大值、min 最小值、（25%、50%、75%）分位数等。







python算法题中一些输入的问题
2020年10月14日 由 samstar
第一个用python做的算法题，输入输出格式真的是千古难题，正好拿来当作例子
 
这题本身没啥难度，主要是输入输出
import sys

hit = []
while True:

    line = sys.stdin.readline() #这三行用来读输入并检测EOF，读取的为字符串
    if not line:
        break

    hit = list(line) #把字符串换成数组
    hit = [x for x in hit if x != ' ' and x != '\n'] #去掉转换出来的多余的东西
    hit.sort(reverse=True) #从大到小排序
    if hit[0] != '0':
        hit = ''.join(hit) #换成字符串输出
        print(hit)
    else:
        print('0')
下面这一段可以直接吧输入的一行变成数组
        hit = list(map(int, input().split()))
        hit.sort(reverse=True)
另一种输入方法
 
这个用了try来找输入的EOF 因该是在 n,m那里，如果输入的不是两个就会进入except
while True:
    try:
        n,m = map(int,input().split())
        l = list(map(int,input().split()))
        l.sort()
        i = 1
        k = 0  # 记录人数
        while i<len(l):
            if l[i]-l[i-1] < m:
                k+=1
                i+=2
            else:
                i+=1
        print(k)
    except:
        break
这样也行，应该是int不能输入回车，所以到了except
while True:
    try:
        n = int(input())
        print(n)
    except:
        break





JAVA用Arrays.sort对数组排序（leetcode）
2020年11月9日 由 samstar
我们有一个由平面上的点组成的列表 points。需要从中找出 K 个距离原点 (0, 0) 最近的点。
（这里，平面上两点之间的距离是欧几里德距离。）
你可以按任何顺序返回答案。除了点坐标的顺序之外，答案确保是唯一的。
输入：points = [[1,3],[-2,2]], K = 1
输出：[[-2,2]]
解释：
(1, 3) 和原点之间的距离为 sqrt(10)，
(-2, 2) 和原点之间的距离为 sqrt(8)，
由于 sqrt(8) < sqrt(10)，(-2, 2) 离原点更近。
我们只需要距离原点最近的 K = 1 个点，所以答案就是 [[-2,2]]。
class Solution {
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        sort(points.begin(), points.end(), [](const vector<int>& u, const vector<int>& v) {
            return u[0] * u[0] + u[1] * u[1] < v[0] * v[0] + v[1] * v[1];
        });
        return {points.begin(), points.begin() + K};
    }
};




JAVA对对象进行排序
2020年11月9日 由 samstar
import java.util.Arrays;
import java.util.Scanner;

public class Main {
    public static class point implements Comparable<point>{
        int dis,i;
        public point(int dis,int i){
            this.dis = dis;
            this.i = i;
        }
        @Override
        public int compareTo(point x) {
            return this.dis-x.dis;
        }
    }

    public static void main(String[] args){
        int dis;
        Scanner input = new Scanner(System.in);
        point [] p = new point[5];
        for(int i=0;i<5;i++){
            dis = input.nextInt();
            point pt = new point(dis,i);
            p[i] = pt;
        }
        Arrays.sort(p);
        for(int i=0;i<5;i++)
            System.out.println(p[i].dis);
    }
}
使用了Comparable接口，实现了用Arrays.sort来对对象数组进行排列


C++关于vector的一些用法
2020年11月16日 由 samstar
总是用错vector，整个文章记一下
对于动态的vector赋值要用.push_back
//将堆栈st顶部的int添加到vecor res的尾部
stack<int> st;
vector<int> ret;

ret.push_back(st.top());

//直接赋值
ret.push_back({i,j});
对vector用sort：
class point{
public:
    int X,Y,NO;
    double dis;
};

bool cmp(point x,point y){
    if(x.dis == y.dis)
        return x.NO < y.NO;
    return x.dis < y.dis;
}
vector<point> p;
sort(p.begin(),p.end(),cmp);
对二维vector用sort：
int main()
{
	vector<vector<int>>ans;
	ans.push_back({ 6,7,8 });
	ans.push_back({ 5,6 });

	sort(ans.begin(), ans.end(), [](vector<int>a, vector<int>b){
             return a[0] < b[0]; 
        });


	return 0;
}
因为所有我们在类内定义的非static成员函数在经过编译后隐式的为他们添加了一个this指针参数,而标准库的sort()函数的第三个cmp函数指针参数中并没有这样this指针参数，因此会出现输入的cmp参数和sort()要求的参数不匹配，从而导致了错误。
因此需要在cmp前加上static

static bool cmp(vector<int> &x,vector<int> &y){
        return x[0] < y[0] || (x[0] == y[0] && x[1] > y[1]);
    }

int maxEnvelopes(vector<vector<int>>& e) {
        sort(e.begin(),e.end(),cmp);
找出vector中的最大值：
int max = *max_element(v.begin(),v.end());
分类目录




tensorflow2.0可视化
2020年12月22日 由 samstar
import os
import tensorflow as tf
from keras.utils import to_categorical
from keras import models, layers
from keras.optimizers import RMSprop
from keras.datasets import mnist

# 设置后台打印日志等级 避免后台打印一些无用的信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 加载数据集，我已经改过load_data函数了
(x, y), (x_tese, y_tese) = mnist.load_data()

# 搭建LeNet网络
def LeNet():
    network = models.Sequential()
    network.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    network.add(layers.MaxPooling2D((2, 2)))
    network.add(layers.Conv2D(16, kernel_size=(5, 5), activation='relu'))
    network.add(layers.MaxPooling2D((2, 2)))
    network.add(layers.Flatten())
    network.add(layers.Dense(120, activation='relu'))
    network.add(layers.Dense(84, activation='relu'))
    network.add(layers.Dense(10, activation='softmax'))
    return network
network = LeNet()

network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

x = x.reshape((60000, 28, 28, 1)).astype('float') / 255
x_tese = x_tese.reshape((10000, 28, 28, 1)).astype('float') / 255
y = to_categorical(y)
y_tese = to_categorical(y_tese)

#可视化
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="fit_logs/", histogram_freq=1)

# 训练                                                    #这里也是可视化
network.fit(x, y, epochs=10, batch_size=128, verbose=2, callbacks=[tensorboard_callback])

#测试
test_loss, test_accuracy = network.evaluate(x_tese, y_tese)
print("test_loss:", test_loss, "    test_accuracy:", test_accuracy)
在控制台里输入下列指令，打开里面的页面即可
tensorboard --logdir="fit_logs/"



pycharm报错crun: error: invalid active developer path ……
2021年2月23日 由 samstar
这个是mac的xcode出问题了一般应该是因为系统更新，重新装一下就行了
xcode-select --install

