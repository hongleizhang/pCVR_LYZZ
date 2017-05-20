#include <iostream>
#include <stdlib.h>
#include <string>
#include <math.h>
#include "matrix.h"
#include <fstream>
#include <sstream>
#include <stack>
using namespace std;
#define MAX 1000000
#define MIN -100000


struct creativeID
{
    int id;
    int sum;
    int ssum;
    double pr;
    struct creativeID *next;
    int len;
};

///统计每个时间段点击激活的概率，直接存储在变量中
double clicktime[24]= {0.0246654,
                       0.0301104,
                       0.028314,
                       0.0229519,
                       0.259183,
                       0.0264976,
                       0.0245146,
                       0.0217758,
                       0.0231793,
                       0.0384133,
                       0.0339939,
                       0.0304794,
                       0.0255811,
                       0.0233522,
                       0.0227103,
                       0.0248416,
                       0.0237889,
                       0.0227326,
                       0.0246128,
                       0.0263791,
                       0.0270367,
                       0.0247819,
                       0.0261875,
                       0.0244826
                      };
///统计不同联网方式点击激活的概率，直接存储在内存
double connection[5]= {0.00479176,
                       0.02966,
                       0.008357,
                       0.00831307,
                       0.00684698
                      };
///统计不同运营商点击激活的概率
double telecoms[4]= {0.0324223,
                     0.0218315,
                     0.0322667,
                     0.0261772
                    };
/**
sigmad函数,这里没有直接将预测类别转换为整型，而是还是返回一个double值
*/
double sigmoid(double z)
{
    return 1.0/(1+exp(-z));
}
/**
梯度下降算法，主要是确定负梯度方向，步长，采用迭代的思想迭代n至收敛，
当目标函数是凸规划问题，那么局部最小值就是全局最小值

**/
int gradAscent(Matrix x,Matrix y)
{
    Matrix weights;
    weights.initMatrix(&weights,x.row,1,1);///初始化优化参数向量为1

    Matrix xT;
    xT.initMatrix(&xT,x.row,x.col);
    xT.transposematrix(x,&xT);

    Matrix z;
    z.initMatrix(&z,x.col,1);

    Matrix w1;
    w1.initMatrix(&w1,x.row,y.row);

    double alpha=0.01;///迭代步长
    double error;///记录错误率
    int k,c=0;
    int i,j;
    double loss=0;
    for(c=0; c<1000000; c++)
    {
        z.multsmatrix(&z,x,weights);

        for(i=0; i<x.col; i++)
            z.mat[i][0]=sigmoid(z.mat[i][0]);///预测类别
        loss=0;
        for(i=0; i<x.col; i++)
        {
            if(y.mat[i][0]==1)
            {
                loss+=log(z.mat[i][0]);
            }
            if(y.mat[i][0]==0)
            {
                loss+=log(1-z.mat[i][0]);
            }
        }
        cout<<"loss="<<loss/z.col<<endl;

        z.submatrix(&z,y,z);///计算负梯度方向，同时可以作为预测类别与真实类别的误差

        error=0;
        for(k=0; k<x.col; k++)///统计错误率
            error+=z.mat[k][0];
        cout<<"error="<<error<<endl;
        if(error<2&&error>-2)///设置错误率小于一定值时退出迭代
            break;
        w1.multsmatrix(&w1,xT,z);///计算负梯度方向
        for(j=0; j<x.row; j++)
            w1.mat[j][0]*=alpha;///负梯度方向与步长的乘积确定迭代值
        weights.addmatrix(&weights,weights,w1);///往负梯度方向走一个步长
        cout<<"weights"<<endl;
        weights.print(weights);
    }
    int er=0;
    for(i=0; i<x.col; i++)
    {
        er+=y.mat[i][0];
    }
    cout<<"label1sum="<<er<<endl;
    /**
    验证算法的正确性
    **/
    double sum=0;
    Matrix test;
    test.initMatrix(&test,y.col,y.row);
    test.multsmatrix(&test,x,weights);
    for(i=0; i<y.col; i++)
    {
        if(test.mat[i][0]>0)
        {
            sum+=1-y.mat[i][0];
        }

        else
        {
            sum+=y.mat[i][0]-0;
        }

    }
    cout<<"sum="<<sum<<endl;
}
/**
随机梯度下降与梯度下降法不同的是在负梯度方向的确定，梯度下降是根据所有的样本来确定负梯度方向，
而随机梯度下降每次只看一个样本点来确定负梯度方向，虽然不完全可信，但随着迭代次数增加，同样收敛

**/
int stoGradAscent(Matrix x,Matrix y)//随机梯度下降每一次选择m个样本进行求梯度下降方向，该代码中只选择一个样本进行求解梯度下降方向与数值
{
    int i,j,c=0;
    Matrix weights;
    weights.initMatrix(&weights,x.row,1,1);

    Matrix z;
    z.initMatrix(&z,1,1);

    Matrix xOne;
    xOne.initMatrix(&xOne,1,x.row);

    Matrix xOneT;
    xOneT.initMatrix(&xOneT,xOne.row,xOne.col);

    Matrix w1;
    w1.initMatrix(&w1,x.row,y.row);

    double alpha=0.01;///步长
    double error;
    double loss;
    for(c=0; c<5000; c++)
    {
        loss=0;
        for(i=0; i<x.col; i++)
        {
            xOne.getOneCol(xOne,x,i);///随机选择一个样本点，这里没有作随机选择，而是按序选择
            z.multsmatrix(&z,xOne,weights);
            z.mat[0][0]=sigmoid(z.mat[0][0]);
            if(y.mat[i][0]==1)
            {
                loss+=log(z.mat[0][0]);
            }
            if(y.mat[i][0]==0)
            {
                loss+=log(1-z.mat[0][0]);
            }
            z.mat[0][0]=y.mat[i][0]-z.mat[0][0];
            xOneT.transposematrix(xOne,&xOneT);
            w1.multsmatrix(&w1,xOneT,z);///根据一样样本的预测误差来确定负梯度方向
            for(j=0; j<w1.row; j++)
                w1.mat[j][0]*=alpha;
            weights.addmatrix(&weights,weights,w1);///迭代

        }
        cout<<"weights"<<endl;
        weights.print(weights);
        cout<<"loss="<<loss/x.col<<endl;
    }
    /**
    验证算法的正确性
    */
    double sum=0;
    Matrix test;
    test.initMatrix(&test,y.col,y.row);
    test.multsmatrix(&test,x,weights);
    for(i=0; i<y.col; i++)
    {
        if(test.mat[i][0]>0)
        {
            sum+=1-y.mat[i][0];
        }

        else
        {
            sum+=y.mat[i][0]-0;
        }

    }
    cout<<"sum="<<sum<<endl;
    cin>>i;
}

/**
逻辑回归，这里主要考虑其常用的两种求参数算法，一种是梯度下降，一种是随机梯度下降

*/

///下面是训练后得到的参数，对应于逻辑回归中的wx+b中的w，b,
///第一个表示clicktime,con,creativeID,userID,positionID,connection,teles
///需要说明的是，由于userID暂时没用，所以对应userID的参数是b
double weights[8]= {-0.445365,0,11.7753,-596.598,17.3998,46.2444,4.75333};

///决策函数，直接对test.csv中的样本进行概率预测，写入pr.csv
int predict(Matrix x,Matrix y)
{
    int i=0;
    ofstream ofile;
    ofile.open("pr.csv");
    for(i=0; i<x.col; i++)
    {
        y.mat[i][0]=x.mat[i][2]*weights[0]+x.mat[i][3]*weights[2]+x.mat[i][5]*weights[4]+x.mat[i][6]*weights[5]+x.mat[i][7]*weights[6]+weights[3]*0.01;
        y.mat[i][0]=sigmoid(y.mat[i][0]);
        ofile<<i+1<<','<<y.mat[i][0]<<'\n';
    }
}


void fect_tj(Matrix x,Matrix y)
{
    ///对clicktime creativeID postionID conenction teles特征进行统计点击激活的概率
    int i,j;
    double sum[5]= {0,0,0,0,0};
    int len[5]= {0,0,0,0,0};
    creativeID *ct;
    ct=new creativeID;
    ct->len=0;
    ct->next=NULL;
    creativeID *p;
    p=ct;
    creativeID *tmpct;

    int tsum[24];
    int l1tsum[24];
    int t=0;


    int snum[10000];
    int ssnum[10000];
    double pr[10000];
    for(i=0; i<10000; i++)
    {
        snum[i]=0;
        ssnum[i]=0;
    }
    for(i=0; i<y.col; i++)
    {
        ssnum[(int)x.mat[i][5]]++;//这里将下标
        {
            snum[(int)x.mat[i][5]]+=y.mat[i][0];
        }
    }

    ofstream ofile;
    ofile.open("positionID.txt");
    for(j=0; j<10000; j++)
    {
        if(snum[j]==0&&ssnum[j]==0)
        {
            pr[j]=0.025;
        }
        if(snum[j]==0&&ssnum[j]!=0)
        {
            pr[j]=1.0/((ssnum[j]+1)*1.0/0.025);
        }
        if(snum[j]!=0&&ssnum[j]==0)
        {
            cout<<"error"<<endl;
            exit(-1);
        }
        if(snum[j]!=0&&ssnum[j]!=0)
        {
            pr[j]=(double)snum[j]/ssnum[j];
        }

        ofile<<j<<"  "<<snum[j]<<"   "<<ssnum[j]<<"   "<<pr[j]<<'\n';
    }
    for(t=0; t<24; t++)
    {
        tsum[t]=0;
        l1tsum[t]=0;
    }

    for(i=0; i<y.col; i++)
    {
        for(t=0; t<24; t++)
        {
            if(((int)x.mat[i][1]%10000)/100==t)
            {
                tsum[t]++;
                if(y.mat[i][0]==1)
                {
                    l1tsum[t]++;
                }
            }
        }
    }


    for(t=0; t<24; t++)
    {
        cout<<tsum[t]<<"&"<<l1tsum[t]<<"&"<<(double)l1tsum[t]/tsum[t]<<endl;
    }

    j=7;//这里6表示connection，7表示teles
    for(i=0; i<y.col; i++)
    {
        {

            if(x.mat[i][j]==0)
            {
                sum[0]+=y.mat[i][0];
                len[0]++;

            }
            if(x.mat[i][j]==1)
            {
                len[1]++;
                sum[1]+=y.mat[i][0];
            }

            if(x.mat[i][j]==2)
            {
                sum[2]+=y.mat[i][0];
                len[2]++;
            }
            if(x.mat[i][j]==3)
            {
                sum[3]+=y.mat[i][0];
                len[3]++;
            }
            if(x.mat[i][j]==4)
            {
                sum[4]+=y.mat[i][0];
                len[4]++;
            }
        }
    }
    for(i=0; i<5; i++)
        cout<<sum[i]<<"  "<<len[i]<<"  "<<(double)sum[i]/len[i]<<endl;
}

int main()
{
    int i,j=0;
    dataToMatrix dtm;
    cout<<"loadData"<<endl;
    cout<<"----------------------"<<endl;
    /**
    对训练集中的样本进行特征映射，生成映射后的特征样本，写到train_fx.txt中，
    同样，对测试集中的样本进行特征映射，生成映射后的特征样本，写到test_fx.txt中
    需要注意，训练样本与测试样本中每列对应的特征不是一致的，所以在写到fx中需要做些改动
    ***/

    dtm.loadData(&dtm,"creativeID.txt",1);
    Matrix cID;
    cID.loadMatrix(&cID,dtm);

    dtm.loadData(&dtm,"positionID.txt",1);
    Matrix pID;
    pID.loadMatrix(&pID,dtm);


    //char file[20]="pre\\test.csv";
    char file[20]="pre\\train.csv";
    dtm.loadData(&dtm,file);

    ///当完成训练样本与测试样本的特征变换之后
    ///就可以把之前注释掉，直接读txt中的样本，进行训练或者测试决策了，读入train_fx.txt表示训练，读入test_fx.txt表示测试，预测实现部分
    ///需要说明的是统计点击转化的概率是读csv文件，直接读200万数据，而实际用于训练时，加载数据的函数在loadData中换了一种方式，只读20万
    ///测试样本的读入，则是全部读入，所以需要去掉loadData.h文件中loadData函数中的if（i%10==2）语句
    //dtm.loadData(&dtm,"train_fx.txt",1);

    //cout<<"ok"<<endl;

    Matrix x;
    x.loadMatrix(&x,dtm);
    cout<<"col="<<x.col<<endl;

    Matrix y;
    y.initMatrix(&y,x.col,1);
    y=y.getOneRow(x,1);

    Matrix fx;
    fx.initMatrix(&fx,x.col,x.row);
    cout<<"col="<<y.col<<endl;
    for(i=0; i<y.col; i++)
    {
        fx.mat[i][1]=clicktime[(int)((int)x.mat[i][1]%10000)/100];
        fx.mat[i][6]=connection[(int)x.mat[i][6]];
        fx.mat[i][7]=telecoms[(int)x.mat[i][7]];
        fx.mat[i][3]=cID.mat[(int)x.mat[i][3]][3];
        fx.mat[i][5]=pID.mat[(int)x.mat[i][5]][3];
        fx.mat[i][4]=0.01;
        fx.mat[i][0]=x.mat[i][0];
        fx.mat[i][2]=0;
    }

    ofstream ofile;
    ofile.open("train_fx.txt");
    for(i=0; i<fx.col; i++)
    {
        for(j=0; j<fx.row; j++)
        {
            ofile<<fx.mat[i][j]<<"  ";
        }
        ofile<<'\n';
    }
    ///训练求参函数实现
    //x.deleteOneRow(&x,1);
    //gradAscent(x,y);
    //stoGradAscent(x,y);


    ///预测测试样本概率函数
    //predict(x,y);
    return 0;
}
