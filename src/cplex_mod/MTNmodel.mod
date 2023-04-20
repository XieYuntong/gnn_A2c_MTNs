/*********************************************
 * OPL 22.1.0.0 Model
 * Author: xieyuntong
 * Creation Date: 2023��2��7�� at ����4:48:38
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
{int} busLine=...; //�����еĹ�����·����
{modRouteTuple} modRouteAttr=...;
{headWayAttrTuple} headWayAttr=...; //���뷢��ʱ�̱�
{demandAttrTuple} demandAttr=...; //������������
{travelAttrTuple} pathAttr = ...; //������<i,j,l>����ʽ��ÿ��������·l;����վ������
{accTuple} acclintTuple=...; //����<i,n>��ʽ����iվ����mod����
{edgeAttrTuple} edgeAttr = ...; //����ڵ�·�����ݣ�������ʻʱ��
{accRLTuple} accRL = ...; //����RL����õ�tʱ��iվ��Ҫmod����
{accRLTuple} desiredModsaddAttr = ...; //����RL����õ���tʱ��l��iվ����Ҫ��ӻ���ٵ�mods����,<i,l,s>
{busTypeTuple} busType =...; //����tʱ�̵ִ�iվ��l��·��mods�������ԣ�<i,l,s>
{busTypeTuple} totalPassengerAttr =...; //����tʱ�̵ִ�iվʱl��·���ϵĳ˿�����
{lineCostTuple} routeCostAttr = ...; //����Ӹ������i������ʻ��·l�Ļ���,key(i,l)
{travelTimeTuple} travelTimeAttr = ...; //����˿ͳ���l�Ź�����i��j��ʱ������,key(i,j,l)
{ReCostTuple} ReCost = ...; //����rebalanbcing�ķ���
float timeCost = 0.2; //ʱ��ɱ�
int modCapacity = 6; //һ��mod������

{modRouteTuple} modRoute = {<i,l> | <i, l> in modRouteAttr}; //����mod��ǲ·������
{edgeTuple} demandEdge = {<i,j>|<i,j,v> in demandAttr}; //�����г�����������յ�����
{travelAttrTuple} travelEdge = {<i,j,l> | <i,j,l> in pathAttr}; //����˿ͳ�����������
{travelAttrTuple} pathEdge = {<i,j,l> | <i,j,l> in pathAttr}; //����������·l�϶�Ӧ��i,jվ��
{edgeTuple} reEdge={<i,j>|<i,j,t> in edgeAttr};
{edgeTuple} Edge={<i,j>|<i,j,l,t> in travelTimeAttr};
{int} line={l|l in busLine};
{int} origin = {i|<i,n> in acclintTuple}; //���������
int accInit[origin] = [i:n|<i,n> in acclintTuple]; //��ȡ���г�������
int demand[demandEdge] = [<i,j>:v|<i,j,v> in demandAttr]; //��ȡ����������
int headway[line] = [l:n|<l,n> in headWayAttr]; //��ȡ��·l�ķ���Ƶ��
float routeCost[modRoute] = [<i, l>:c | <i, l, c> in routeCostAttr];
int travelTime[pathEdge] = [<i, j, l>:t | <i, j, l, t> in travelTimeAttr]; //�˿ͳ���l�Ź�����i��j��ʱ������,key(i,j,l)
int modsNum[modRoute] = [<i,l>:s|<i,l,s> in busType]; //��ȡtʱ�̵���iվ��l�߹�����mods������
int desiredModsadd[origin] = [i:n|<i,n> in desiredModsaddAttr]; //iվ����Ҫ��ӻ���ٵ�l��·�ϵ�mods����
int totalPassenger[modRoute] = [<i, l>:n|<i,l,n> in totalPassengerAttr]; //��t-1ʱ��l��·��iվ�������ܳ˿�����,key(i,l)
//�ж�i,j�Ƿ���busline l��
float reCost[reEdge] = [<i,j>:c | <i,j,c> in ReCost];

float desiredVehicles[origin] = [i:n|<i,n> in accRL]; //��ȡRL����õ���ʱ��t��i��Ҫ�ĳ����� 

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

  
  


















