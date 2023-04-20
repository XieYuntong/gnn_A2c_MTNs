/*********************************************
 * OPL 22.1.0.0 Model
 * Author: xieyuntong
 * Creation Date: 2023年2月7日 at 下午4:48:38
 *********************************************/
tuple demandAttrTuple{
  int i;
  int j;
  int v;
}

tuple travelAttrTuple{
  int i;
  int j;
  int l;
}

tuple travelTimeTuple{
  int i;
  int j;
  int l;
  int t;
}

tuple modRouteTuple{
  int i;
  int l;
}

tuple accTuple{
  int i;
  int n;
}

tuple edgeAttrTuple{
  int i;
  int j;
  int t;
}

tuple edgeTuple{
  int i;
  int j;
}

tuple accRLTuple{
  int i;
  int n;
}

tuple headWayAttrTuple{
  int l;
  int t;
}

tuple busTypeTuple{
  int i;
  int l;
  int s;
}

tuple lineCostTuple{
  int i;
  int l;
  float c;
}

tuple ReCostTuple{
  int i;
  int j;
  float c;
}

string path = ...;
{int} busLine=...; //将已有的公交线路输入
{modRouteTuple} modRouteAttr=...;
{headWayAttrTuple} headWayAttr=...; //输入发车时刻表
{demandAttrTuple} demandAttr=...; //输入需求数据
{travelAttrTuple} pathAttr = ...; //以数组<i,j,l>的形式将每条公交线路l途经的站点输入
{accTuple} acclintTuple=...; //数组<i,n>形式输入i站现有mod数量
{edgeAttrTuple} edgeAttr = ...; //输入节点路段数据，包括行驶时间
{accRLTuple} accRL = ...; //输入RL计算得到t时刻i站需要mod数量
{accRLTuple} desiredModsaddAttr = ...; //输入RL计算得到的t时刻l在i站所需要添加或减少的mods数量,<i,l,s>
{busTypeTuple} busType =...; //输入t时刻抵达i站的l线路的mods车队属性，<i,l,s>
{busTypeTuple} totalPassengerAttr =...; //输入t时刻抵达i站时l线路车上的乘客总量
{lineCostTuple} routeCostAttr = ...; //输入从各个起点i出发行驶线路l的花费,key(i,l)
{travelTimeTuple} travelTimeAttr = ...; //输入乘客乘坐l号公交从i到j的时间数据,key(i,j,l)
{ReCostTuple} ReCost = ...; //输入rebalanbcing的费用
float timeCost = 0.2; //时间成本
int modCapacity = 6; //一个mod的容量

{modRouteTuple} modRoute = {<i,l> | <i, l> in modRouteAttr}; //定义mod派遣路线数组
{edgeTuple} demandEdge = {<i,j>|<i,j,v> in demandAttr}; //定义有出行需求的起终点数组
{travelAttrTuple} travelEdge = {<i,j,l> | <i,j,l> in pathAttr}; //定义乘客出行特征数组
{travelAttrTuple} pathEdge = {<i,j,l> | <i,j,l> in pathAttr}; //定义所有线路l上对应的i,j站点
{edgeTuple} reEdge={<i,j>|<i,j,t> in edgeAttr};
{edgeTuple} Edge={<i,j>|<i,j,l,t> in travelTimeAttr};
{int} line={l|l in busLine};
{int} origin = {i|<i,n> in acclintTuple}; //定义出发地
int accInit[origin] = [i:n|<i,n> in acclintTuple]; //读取现有车辆数据
int demand[demandEdge] = [<i,j>:v|<i,j,v> in demandAttr]; //读取需求量数据
int headway[line] = [l:n|<l,n> in headWayAttr]; //读取线路l的发车频率
float routeCost[modRoute] = [<i, l>:c | <i, l, c> in routeCostAttr];
int travelTime[pathEdge] = [<i, j, l>:t | <i, j, l, t> in travelTimeAttr]; //乘客乘坐l号公交从i到j的时间数据,key(i,j,l)
int modsNum[modRoute] = [<i,l>:s|<i,l,s> in busType]; //读取t时刻到达i站的l线公交的mods的数量
int desiredModsadd[origin] = [i:n|<i,n> in desiredModsaddAttr]; //i站所需要添加或减少的l线路上的mods数量
int totalPassenger[modRoute] = [<i, l>:n|<i,l,n> in totalPassengerAttr]; //在t-1时刻l线路从i站出发的总乘客人数,key(i,l)
//判断i,j是否在busline l上
float reCost[reEdge] = [<i,j>:c | <i,j,c> in ReCost];

float desiredVehicles[origin] = [i:n|<i,n> in accRL]; //读取RL计算得到的时间t中i需要的车辆数 

dvar int vehicleFlow[modRoute];
dvar int+ rebFlow[reEdge];
dvar int+ passengerFlow[travelEdge];
dvar boolean Alpha[modRoute];

minimize
  sum(e in modRoute)(vehicleFlow[e] * routeCost[e])+
  sum(e in reEdge)(rebFlow[e] * reCost[e])+
  sum(e in travelEdge)(timeCost * passengerFlow[e] * (0.5 * headway[e.l]))+
  sum(i in origin)(abs(sum(e in modRoute: e.i==i)(vehicleFlow[<e.i,e.l>])-desiredModsadd[i]))+
  sum(e in Edge)(abs(demand[e] - sum(m in travelEdge: m.i==e.i && m.j==e.j)(passengerFlow[<m.i, m.j, m.l>])));
  
subject to{
   
     forall(e in modRoute)
        modCapacity*vehicleFlow[e] >= (totalPassenger[e] + (sum(m in travelEdge: m.i==e.i && m.l==e.l&& m.i!=m.j)(passengerFlow[<m.i, m.j, m.l>]))-modCapacity*modsNum[e]);
     
          forall(e in modRoute)
              vehicleFlow[e] <= 100000 * Alpha[e];              
                
               forall(i in origin)
                  sum(e in modRoute: e.i==i)(vehicleFlow[<e.i, e.l>]) + sum(e in reEdge: e.i==i && e.i!=e.j)(rebFlow[<e.i, e.j>]) <= accInit[i];
                  
                  forall(e in Edge)
                    demand[e] >= sum(m in travelEdge: m.i==e.i && m.j==e.j)(passengerFlow[<m.i, m.j, m.l>]);
                    
                    forall(i in origin)
                      sum(e in reEdge: e.i==i && e.i!=e.j) (rebFlow[<e.j, e.i>] - rebFlow[<e.i, e.j>]) >= desiredVehicles[i] - accInit[i];
                                   
}     

main {
  thisOplModel.generate();
  cplex.solve();
  var ofile = new IloOplOutputFile(thisOplModel.path);
  ofile.write("vehicleflow=[")
  for(var e in thisOplModel.modRoute)
       {
         ofile.write("(");
         ofile.write(e.i);
         ofile.write(",");
         ofile.write(e.l);
         ofile.write(",");
         ofile.write(thisOplModel.vehicleFlow[e]);
         ofile.write(")");
       }
  ofile.writeln("];")
  
  ofile.write("rebflow=[")
  for(var e in thisOplModel.reEdge)
       {
         ofile.write("(");
         ofile.write(e.i);
         ofile.write(",");
         ofile.write(e.j);
         ofile.write(",");
         ofile.write(thisOplModel.rebFlow[e]);
         ofile.write(")");
         }
  ofile.writeln("];")
  
  ofile.write("Alpha=[")
  for(var e in thisOplModel.modRoute)
       {
         ofile.write("(");
         ofile.write(e.i);
         ofile.write(",");
         ofile.write(e.l);
         ofile.write(",");
         ofile.write(thisOplModel.Alpha[e]);
         ofile.write(")");
         }
   ofile.writeln("];")
   
  ofile.write("passengerflow=[")
  for(var e in thisOplModel.travelEdge)
       {
         ofile.write("(");
         ofile.write(e.i);
         ofile.write(",");
         ofile.write(e.j);
         ofile.write(",");
         ofile.write(e.l);
         ofile.write(",");
         ofile.write(thisOplModel.passengerFlow[e]);
         ofile.write(")");
         }
  ofile.writeln("];") 
  ofile.close();
}

  
  


















