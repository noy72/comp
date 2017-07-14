#treeDP
# 全方位木DPによる木の直径の演算
#closedLoop
# 閉路の検出
#JoinInterval
# 区間の結合
#repalceAll
# 文字列の置き換え
#segTree
# セグメントツリークラス
#eulerPhi
# オイラー関数
#extgcd
# 拡張ユークリッドの互除法
#simultaneousLinearEquations
# 連立一次方程式
#flow
# 最大流
#matrix
# 行列計算
#meet_in_the_middle
# bitによる全通列挙
#compress coordinate
# 座標圧縮
#lowest common ancestor - doubling
# ダブリングを利用したLCA
#gridUnion-find
# グリッドグラフのユニオン木
#dp_Partial_sum_with_number_restriction
# 個数制限付き部分和
#pascals_triangle
# n個を選ぶ組み合わせの中、k個を選ぶ組み合わせの割合。
#intervalState
# 区間の関係
#toDAG
# 2点間の最短経路になる辺を残したDAG
#combination
# べき乗/階乗/コンビネーション
#ternarySearch
# 三分探索
#numOfnumber
# nまでの数字を書いたとき、1が出現する回数
#next_combination
# 組み合わせの全列挙
#eightQueensProblem
# 8クイーン問題
#numPrimeFactor
# 約数の個数を求める
#bitManip
# ビットの操作
#oneStrokePath
# 全ての頂点を通る一筆書きの総数
#cumulativeSum
# 2次元累積和
#levenshteinDistance
# 編集距離を求めるdp
#areaOfPolygon
# 凹みのない多角形の面積を求める
#areaOfTriangle
# 三角形の面積を求める
#bfsOfAdjacencyMatrix
# 隣接行列の幅優先探索
#bfsOfGrid
# グリッドの幅優先探索
#dfsOfTree
# 木の深さ優先探索
#binarySearch
# 二分探索
#geometry
# 幾何、凸包
#eratosthenes
# 10^6以下の素数を全列挙
#GreatestCommonDivisor
# 最大公約数
#LowestCommonMultiple
# 最小公倍数
#leapYear
# うるう年判定
#maze
# 迷路をグリッドに拡張
#nextMonth
# 日付の計算
#parser
# 構文解析
#power
# 冪乗
#rotate
# 行列の回転
#toNum
# 文字列から数値への変換
#toStr
# 数値から文字列への変換
#union-find
# 森
#warchallFloyd
# 全点最短経路
#dijkstraOfAdjacencyList
# 隣接リストのダイクストラ
#bellmanFord
# 負の経路を含む最短経路、負の経路の検出
#prim
# 最小無向全域木
#binaryIndexedTree
# セグメント木、区間の和
#LCS
# 最長共通部分文字列の長さ
#LIS
# 最長増加部分列
#divisor
# 約数の列挙
#primeVactor
# 素因数分解
#syakutori
# しゃくとり
#isUniqueStr
# 文字列の文字がユニークか判定
#areAnagram
# 文字列がアナグラムかを判定
#spilt
# 文字列を空白で区切る
#articulationPoints
# 無向グラフのの関節点、橋を全列挙
#stronglyConnectedComponents 
# 強連結性成分分解
#topologicalSort
# トポロジカルソート
#diameterOfTree
# 木の直径を求める
#heightOfTree
# 木の高さを求める
#minCostFlow
# 最小費用流
#bipartiteMatching
# 2部マッチング



snippet vector
abbr    vector<>()
    vector<${1}> ${0}

snippet pair
abbr    pair<,>
    pair<${1}, ${2}>${0}

snippet     mod
abbr        10^9 + 7
options     head
    const int M = 1000000007;

snippet     inf
abbr        const int INF = ;
options     head
    const int INF = ${1};${0}

snippet     cinup
abbr        cin高速化
options     head
    cin.tie(0);
    ios::sync_with_stdio(false);↲

snippet     innn
abbr        int n; cin >> n;
options     head
    int n;
    cin >> n;

snippet     renn
abbr        rep(i,n){
options     head
    rep(i,n){

    }

snippet     upb
abbr        二分探索
    upper_bound(all(${1}), ${2});${0}

snippet     lwb
abbr        二分探索
    lower_bound(all(${1}), ${2});${0}

snippet     dydx
abbr        座標移動の配列
options     head
    const int dy[16] = { 0,-1, 0, 1, 1,-1, 1,-1, 0,-2, 0, 2};
    const int dx[16] = { 1, 0,-1, 0, 1, 1,-1,-1, 2, 0,-2, 0};

#凹みのない多角形の面積を求める
snippet areaOfPolygon
abbr areaOfPolygon

 typedef struct {
    double x, y;
 }point;

 double AreaOfTriangle(point a, point b, point c){
    b.x-=a.x;
    b.y-=a.y;
    c.x-=a.x;
    c.y-=a.y;
    return abs((b.x * c.y - b.y * c.x) / 2);
 }

 double areaOfPolygon(point point[/*N*/]){
    int i = 0, second = 0, third = 1;
    double area = 0;
    rep(j,i - 2){
        if(second < third) second+=2;
        else third+=2;
        area+=AreaOfTriangle(point[0], point[second], point[third]);
    }
    return area;
 }

#三角形の面積を求める
snippet areaOfTriangle
abbr areaOfTriangle

 typedef struct {
    double x, y;
 }point;

 double AreaOfTriangle(point a, point b, point c){
    b.x-=a.x;
    b.y-=a.y;
    c.x-=a.x;
    c.y-=a.y;
    return abs((b.x * c.y - b.y * c.x) / 2);
 }

#幅優先探索
snippet bfsOfAdjacencyMatrix
abbr 隣接行列の幅優先探索

 const int N = ;

 int M[N][N];

 void bfs(int n){
    int dis[N]; //距離
    queue<int> q; //訪問した点を入れる
    rep(i,N) dis[i] = INF;

    dis[1] = 0;
    q.push(1);

    int u;
    while(!q.empty()){
        u = q.front(); q.pop();
        rep(v,n + 1){
            if(M[u][v] && dis[v] == INF){
                dis[v] = dis[u] + 1; //グラフの深さ 
                q.push(v);
            }
        }
    }
 }

snippet     bfsOfGrid
abbr        グリッドの幅優先探索

 const int N;
 
 struct Point{ int x, y; };
 int dy[4] = {0,1,0,-1}, dx[4] = {1,0,-1,0};
 bool M[N][N];
 
 int bfs(int h, int w, Point p){
     int dis[N][N];
     queue<Point> q;
     rep(i,N) rep(j,N) dis[i][j] = INF;
 
     dis[p.y][p.x] = 0;
     q.push(p);
 
     Point u;
     while(not q.empty()){
         u = q.front(); q.pop();
         rep(i,4){
             Point next;
             next.x = u.x + dx[i];
             next.y = u.y + dy[i];
             if(next.x < 0 || next.x >= w || next.y < 0 || next.y >= h) continue;
             if(dis[next.y][next.x] == INF && M[next.y][next.x]){
                 dis[next.y][next.x] = dis[u.y][u.x] + 1;
                 q.push(next);
             }
         }
     }
     return /*返り値*/;
 }


snippet     dfsOfTree
abbr        木構造の深さ優先探索
options     head

 typedef struct{
     int parent, left, right;
 } Node;
 Node t[1002];
 
 void dfs(int u, int d){
     if(t[u].left != INF){
         dfs(m, t[u].left, d + 1);
     }
     if(t[u].right != INF){
         dfs(m, t[u].right, d);
     }
 }
 
 int brotherNum(int u){
     if(t[u].right == INF){
         return u;
     }else{
         return brotherNum(t[u].right);
     }
 }
 
 void inputData(int par){
     t[i + 1].parent = par;
     if(t[par].left == INF){
         t[par].left = i + 1;
     }else{
         t[brotherNum(t[par].left)].right = i + 1;
     }
 }
 
 void printGraph(){
     range(i,1,n + 1){ cout << t[i].parent << ' ' << t[i].left << ' ' << t[i].right << endl; }
 }

snippet     binarySearch
abbr        二分探索

 int right = , left = ;
 rep(i,100){
     int mid = (right + left ) / 2;
     if(C(mid)) right = mid;
     else left = mid;
 }

#幾何
snippet geometry
abbr 幾何

 typedef complex<double> Point;
 typedef Point Vector;
 //線分を表す構造体
 struct Segment{ Point p1, p2; };
 //直線を表す構造体
 typedef Segment Line;
 //多角形を表す構造体
 typedef vector<Point> Polygon;
 
 namespace std{
     bool operator < (const Point &a, const Point &b){
         return real(a) != real(b) ? real(a) < real(b) : imag(a) < imag(b);
     }
     bool operator == (const Point &a, const Point &b){
         return a.real() == b.real() && a.imag() == b.imag();
     }
 }
 
 class Circle{
     public:
         Point c;
         double r;
         Circle(Point c = Point(), double r = 0.0): c(c), r(r) {}
 };
 
 // 許容する誤差
 #define EPS (1e-10)
 
 // ベクトルaの絶対値を求める
 //double length = abs(a);
 
 // 2点a,b間の距離を求める
 //double distance = abs(a-b);
 
 /*
 // ベクトルaの単位ベクトルを求める
 Point b = a / abs(a);
 
 // ベクトルaの法線ベクトルn1,n2を求める
 Point n1 = a * Point(0, 1);
 Point n2 = a * Point(0, -1);
 */
 
 int ccw(Point, Point, Point);
 
 // 2つのスカラーが等しいかどうか
 bool EQ(double a, double b){
     return (abs(a - b) < EPS);
 }
 
 // 2つのベクトルが等しいかどうか
 bool EQV(Vector a, Vector b){
     return ( EQ(a.real(), b.real()) && EQ(a.imag(), b.imag()) );
 }
 
 // 内積 (dot product) : a・b = |a||b|cosΘ
 double dot(Point a, Point b) {
     return (a.real() * b.real() + a.imag() * b.imag());
 }
 
 // 外積 (cross product) : a×b = |a||b|sinΘ
 double cross(Point a, Point b) {
     return (a.real() * b.imag() - a.imag() * b.real());
 }
 
 // 2直線の直交判定 : a⊥b <=> dot(a, b) = 0
 bool isOrthogonal(Point a1, Point a2, Point b1, Point b2) {
     return EQ( dot(a1-a2, b1-b2), 0.0 );
 }
 bool isOrthogonal(Line s1, Line s2) {
     return isOrthogonal(s1.p1, s1.p2, s2.p1, s2.p2);
 }
 
 // 2直線の平行判定 : a//b <=> cross(a, b) = 0
 bool isParallel(Point a1, Point a2, Point b1, Point b2) {
     return EQ( cross(a1-a2, b1-b2), 0.0 );
 }
 bool isParallel(Line s1, Line s2) {
     return isParallel(s1.p1, s1.p2, s2.p1, s2.p2);
 }
 
 // 点cが直線a,b上にあるかないか
 bool isPointOnLine(Point a, Point b, Point c) {
     return EQ( cross(b-a, c-a), 0.0 );
 }
 bool isPointOnLine(Line s, Point c) {
     return isPointOnLine(s.p1, s.p2, c);
 }
 
 // 点a,bを通る直線と点cとの距離
 double distanceLPoint(Point a, Point b, Point c) {
     return abs(cross(b-a, c-a)) / abs(b-a);
 }
 double distanceLPoint(Line s, Point c) {
     return distanceLPoint(s.p1, s.p2, c);
 }
 
 // 点a,bを端点とする線分と点cとの距離
 double distanceLsPoint(Point a, Point b, Point c) {
     if ( dot(b-a, c-a) < EPS ) return abs(c-a);
     if ( dot(a-b, c-b) < EPS ) return abs(c-b);
     return abs(cross(b-a, c-a)) / abs(b-a);
 }
 double distanceLsPoint(Segment s, Point c) {
     return distanceLsPoint(s.p1, s.p2, c);
 }
 
 // a1,a2を端点とする線分とb1,b2を端点とする線分の交差判定
 bool isIntersectedLs(Point a1, Point a2, Point b1, Point b2) {
     return ( ccw(a1, a2, b1) * ccw(a1, a2, b2) <= 0 &&
             ccw(b1, b2, a1) * ccw(b1, b2, a2) <= 0 );
 }
 bool isIntersectedLs(Segment s1, Segment s2) {
     return isIntersectedLs(s1.p1, s1.p2, s2.p1, s2.p2);
 }
 
 // a1,a2を端点とする線分とb1,b2を端点とする線分の交点計算
 Point intersectionLs(Point a1, Point a2, Point b1, Point b2) {
     Vector base = b2 - b1;
     double d1 = abs(cross(base, a1 - b1));
     double d2 = abs(cross(base, a2 - b1));
     double t = d1 / (d1 + d2);
 
     return Point(a1 + (a2 - a1) * t);
 }
 Point intersectionLs(Segment s1, Segment s2) {
     return intersectionLs(s1.p1, s1.p2, s2.p1, s2.p2);
 }
 
 // a1,a2を通る直線とb1,b2を通る直線の交差判定
 bool isIntersectedL(Point a1, Point a2, Point b1, Point b2) {
     return !EQ( cross(a1-a2, b1-b2), 0.0 );
 }
 bool isIntersectedL(Line l1, Line l2) {
     return isIntersectedL(l1.p1, l1.p2, l2.p1, l2.p2);
 }
 
 // a1,a2を通る直線とb1,b2を通る直線の交点計算
 Point intersectionL(Point a1, Point a2, Point b1, Point b2) {
     Point a = a2 - a1; Point b = b2 - b1;
     return a1 + a * cross(b, b1-a1) / cross(b, a);
 }
 Point intersectionL(Line l1, Line l2) {
     return intersectionL(l1.p1, l1.p2, l2.p1, l2.p2);
 }
 
 // 線分s1と線分s2の距離
 double distanceLL(Segment s1, Segment s2){
     if(isIntersectedLs(s1.p1, s1.p2, s2.p1, s2.p2) ) return 0.0;
     return min(
             min(distanceLsPoint(s1.p1, s1.p2, s2.p1),
                 distanceLsPoint(s1.p1, s1.p2, s2.p2)),
             min(distanceLsPoint(s2.p1, s2.p2, s1.p1),
                 distanceLsPoint(s2.p1, s2.p2, s1.p2)) );
 }
 double distanceLL(Point p0, Point p1, Point p2, Point p3){
     Segment s1 = Segment{p0, p1}, s2 = Segment{p2, p3};
     return distanceLL(s1, s2);
 }
 
 // 線分sに対する点pの射影
 Point project(Segment s, Point p){
     Vector base = s.p2 - s.p1;
     double r = dot(p - s.p1, base) / norm(base);
     return Point(s.p1 + base * r);
 }
 
 //線分sを対象軸とした点pの線対称の点
 Point reflect(Segment s, Point p){
     return Point(p + (project(s, p) - p) * 2.0);
 }
 
 //点pをangle分だけ時計回りに回転
 Point rotation(Point p, double angle){
     double x, y;
     x = p.real() * cos(angle) - p.imag() * sin(angle);
     y = p.real() * sin(angle) + p.imag() * cos(angle);
     return Point(x, y);
 }
 
 //円cと線分lの交点
 pair<Point, Point> getCrossPoints(Circle c, Line l){
     Vector pr = project(l, c.c);
     Vector e = (l.p2 - l.p1) / abs(l.p2 - l.p1);
     double base = sqrt(c.r * c.r - norm(pr - c.c));
     return make_pair(pr + e * base, pr - e * base);
 }
 
 //円c1と円c2の交点
 double arg(Vector p) { return atan2(p.imag(), p.real()); }
 Vector polar(double a, double r) { return Point(cos(r) * a, sin(r) *a); }
 
 pair<Point, Point> getCrossPoints(Circle c1, Circle c2){
     double d = abs(c1.c - c2.c);
     double a = acos((c1.r * c1.r + d * d - c2.r * c2.r) / (2 * c1.r * d));
     double t = arg(c2.c - c1.c);
     return make_pair(Point(c1.c + polar(c1.r, t + a)), Point(c1.c + polar(c1.r, t - a)));
 }
 
 //点の内包
 static const int IN = 2;
 static const int ON = 1;
 static const int OUT = 0;
 
 int contains(Polygon g, Point p){
     int n = g.size();
     bool x = false;
     rep(i,n){
         Point a = g[i] - p, b = g[(i + 1) % n] - p;
         if( abs(cross(a, b)) < EPS && dot(a,  b) < EPS ) return ON;
         if( a.imag() > b.imag() ) swap(a, b);
         if( a.imag() < EPS && EPS < b.imag() && cross(a, b) > EPS ) x = not x;
     }
     return ( x ? IN : OUT );
 }
 
 //ベクトルの位置検出
 static const int COUNTER_CLOCKWISE = 1;
 static const int CLOCKWISE = -1;
 static const int ONLINE_BACK = 2;
 static const int ONLINE_FRONT = -2;
 static const int ON_SEGMENT = 0;
 
 int ccw(Point p0, Point p1, Point p2){
     Vector a = p1 - p0;
     Vector b = p2 - p0;
     if( cross(a, b) > EPS ) return COUNTER_CLOCKWISE;
     if( cross(a, b) < -EPS ) return CLOCKWISE;
     if( dot(a, b) < -EPS ) return ONLINE_BACK;
     if( abs(a) < abs(b) ) return ONLINE_FRONT;
 
     return ON_SEGMENT;
 }
 
 //凸包
 Polygon convexHull( Polygon s ){
     Polygon u;
     if( s.size() < 3 ) return s;
     sort(s.begin(), s.end());
 
     range(i,0,s.size()){
         //== COUNTER_CLOCKWISEだと内角は180以下（一直線上に並んでいても、頂点として数える）
         //!= CLOCKWISEだと内角は180未満（一直線上の頂点は数えない）
         for(int n = u.size(); n >= 2 && ccw(u[n-2], u[n-1], s[i]) == COUNTER_CLOCKWISE; n--){
             u.pop_back();
         }
         u.emplace_back(s[i]);
     }
 
     for(int i = s.size() - 2; i >= 0; i--){
         //ここも == と != を変更する
         for(int n = u.size(); n >= 2 && ccw(u[n-2], u[n-1], s[i]) == COUNTER_CLOCKWISE; n--){
             u.pop_back();
         }
         u.emplace_back(s[i]);
     }
 
     reverse(u.begin(), u.end());
     u.pop_back();
 
     //最も下にある点の中で最も右にある点から反時計回りに並び替え
     /*
     int i = 0;
     while(i < u.size() - 1){
         if(u[i].imag() > u[i + 1].imag()){
             u.emplace_back(u[i]);
             u.erase(u.begin());
             continue;
         }else if(u[i].imag() == u[i + 1].imag() && u[i].real() > u[i + 1].real()){
             u.emplace_back(u[i]);
             u.erase(u.begin());
             continue;
         }
         break;
     }
     */
 
     return u;
 }
 
 //キャリパー法を用いて凸多角形の直径を求める
 double diameterOfConvexPolygon(Polygon p){
     Polygon s = convexHull(p);
     int n = s.size();
 
     if(n == 2) return abs(s[1] - s[0]);
 
     int i = 0, j = 0;
     rep(k,n){
         if(not (s[i] < s[k])) i = k;
         if(s[j] < s[k]) j = k;
     }
 
     double ret = 0.0;
     int is = i, js = j;
 
     while(i != js || j != is){
         ret = max(ret, abs(s[i] - s[j]));
         if(cross(s[(i + 1) % n] - s[i], s[(j + 1) % n] - s[j]) < 0){
             i = (i + 1) % n;
         }else{
             j = (j + 1) % n;
         }
     }
     return ret;
 }
 
 //凸多角形の切り取りに使う関数。これがなんなのかはまだ知らない。
 Point getCrossPointLL(Line a, Line b){
     double A = cross(a.p2 - a.p1, b.p2 - b.p1);
     double B = cross(a.p2 - a.p1, a.p2 - b.p1);
     if(abs(A) < EPS && abs(B) < EPS) return b.p1;
     return b.p1 + (b.p2 - b.p1) * (B / A);
 }
 
 Polygon convexCut(Polygon p, Line l) {
     Polygon q;
     rep(i,p.size()){
         Point a = p[i], b = p[(i + 1) % p.size()];
         if (ccw(l.p1, l.p2, a) != -1) q.emplace_back(a);
         if (ccw(l.p1, l.p2, a) * ccw(l.p1, l.p2, b) < 0){
             q.emplace_back(getCrossPointLL(Line{a, b}, l));
         }
     }
     return q;
 }
 
 //三角形の面積
 double AreaOfTriangle(Point a, Point b, Point c){
     double w, x, y, z;
     w = b.real()-a.real();
     x = b.imag()-a.imag();
     y = c.real()-a.real();
     z = c.imag()-a.imag();
     return abs((w * z - x * y) / 2);
 }
 
 //多角形の面積
 double areaOfPolygon(Polygon g){
     int n = g.size();
     double ret = 0.0;
     rep(i,n) ret += cross(g[i], g[ (i + 1) % n ]);
     return abs(ret) / 2.0;
 }
 
 //凸多角形かどうかの判定
 bool isConvex(Polygon g){
     int n = g.size();
     rep(i,n){
         if(ccw(g[i], g[(i + 1) % n], g[(i + 2) % n]) == CLOCKWISE) return false;
     }
     return true;
 }

 //凹多角形を線分lで切断した際の多角形の数
 int dividedPolygonNumber(Polygon p, Line l){
     int cnt = 0;
         rep(i,p.size()){
                 if(isIntersectedLs(p[i], p[(i + 1) % p.size()], l.p1, l.p2)) cnt++;
         }
     return cnt / 2 + 1;
 }

 //多角形が点対象となる点の座標
 Point pointSymmetry(Polygon g){
     int size = g.size() / 2;
     if(g.size() % 2) return Point{INF,INF};
 
     set<Point> s;
     rep(i,size){
         rep(j,size){
             if(i == j) continue;
             s.insert(intersectionLs(g[i], g[i + size], g[j], g[j + size]));
         }
     }
     if(s.size() > 1) return Point{INF,INF};
     return *s.begin();
 }

snippet     eratosthenes
abbr        10^6以下の素数を全列挙
options     head
    
 const int kN;
 void primeNumber(bool prime[kN]){
     rep(i,kN) prime[i] = 1;
     prime[0] = prime[1] = 0;
     rep(i,kN){
         if(prime[i]){
             for(int j = i + i; j < kN; j+=i){
                 prime[j] = 0;
             }
         }
     }
 }

snippet     GreatestCommonDividor
abbr        最大公約数
options     head
    
 int gcd(int x, int y) {
    int r;
    if(x < y) swap(x, y);

    while(y > 0){
        r = x % y;
        x = y;
        y = r;
    }
    return x;
 }

snippet     LowestCommonMultiple
abbr        最小公倍数
options     head

 int gcd(int x, int y) {
    int r;
    if(x < y) swap(x, y);

    while(y > 0){
        r = x % y;
        x = y;
        y = r;
    }
    return x;
 }
    
 int lcm( int m, int n ) {
    // 引数に０がある場合は０を返す
    if ( ( 0 == m ) || ( 0 == n ) ) return 0;
    return ((m / gcd(m, n)) * n); // lcm = m * n / gcd(m,n)
 }

snippet     isPrime
abbr        素数判定
options     head
    
 bool primeNumber(int n){
    if(n < 2) return 0;
    else{
        for(int i = 2; i * i <= n; i++){
            if(n % i == 0) return 0;
        }
        return 1;
    }
 }

snippet     leapYear
abbr        うるう年判定
options     head
    
 /*
 うるう年判定
 400で割り切れる年数から数えて、0-399年間でうるう年は97回
 */
 bool leapYear(int y){
     if(y % 400 == 0 || (y % 4 == 0 && y % 100 != 0 )) return true;↲
     else return false;
 }

snippet     maze
abbr        迷路を二次平面上に拡張する
options     head
    
 const int N = ;

 void printMaze(int w, int h, bool M[N][N]){
    rep(i,h + h - 1){
        rep(j,w + w + 1){
            cout << M[i][j];
        }
        cout << endl;
    }
 }

 void extensionOfMaze(int w, int h, bool M[N][N]){
    rep(i,N) rep(j,N) M[i][j] = 1;
    rep(i,h + h - 1){
        if(i % 2){ //横線
            for(int j = 0; j <= w + w; j++){
                if(j == 0 || j == w + w) M[i + 1][j + 1] = 1;//壁
                else if(j % 2 == 0) M[i + 1][j + 1] = 1;
                else cin >> M[i + 1][j + 1];
            }
        }else{ //縦線
            for(int j = 0; j <= w + w; j++){
                if(j == 0 || j == w + w) M[i + 1][j + 1] = 1;//壁
                else if(j % 2 == 1) M[i + 1][j + 1] = 0;
                else cin >> M[i + 1][j + 1];
            }
        }
    }
 }

snippet     nextMonth
abbr        日付の計算
options     head

 bool isLeapYear(int y){
     if(y % 400 == 0 || (y % 4 == 0 && y % 100 != 0 )) return true;↲
     else return false;
 }
    
 void nextMonth(int &y, int &m, int &d){
    bool leapYear = isLeapYear(y);
    if((m == 2 && d == 30 && leapYear) ||
       (m == 2 && d == 29 && !leapYear) ||
       ((m == 4 || m == 6 || m == 9 || m == 11) && d == 31) ||
       ((m == 1 || m == 3 || m == 5 || m == 7 || m == 8 || m == 10 || m == 12) && d == 32)){
        m++;
        d = 1;
    }
    if(m == 13){
        y++;
        m = 1;
    }
 }

snippet     parser
abbr        構文解析
options     head
    
 typedef string::const_iterator State;
 int number(State&);
 int factor(State&);
 int term(State&);
 int expression(State&);
 
 // 数字の列をパースして、その数を返す。
 int number(State &begin) {
     int ret = 0;
 
     while (isdigit(*begin)) {
         ret *= 10;
         ret += *begin - '0';
         begin++;
     }
 
     return ret;
 }
 
 // 括弧か数をパースして、その評価結果を返す。
 int factor(State &begin) {
     if (*begin == '(') {
         begin++; // '('を飛ばす。
         int ret = expression(begin);
         begin++; // ')'を飛ばす。
         return ret;
     } else {
         return number(begin);
     }
     return 0;
 }
 
 // 乗算除算の式をパースして、その評価結果を返す。
 int term(State &begin) {
     int ret = factor(begin);
 
     for (;;) {
         if (*begin == '*') {
             begin++;
             ret *= factor(begin);
         } else if (*begin == '/') {
             begin++;
             ret /= factor(begin);
         } else {
             break;
         }
     }
 
     return ret;
 }
 
 // 四則演算の式をパースして、その評価結果を返す。
 int expression(State &begin) {
     int ret = term(begin);
 
     for (;;) {
         if (*begin == '+') {
             begin++;
             ret += term(begin);
         } else if (*begin == '-') {
             begin++;
             ret -= term(begin);
         } else {
             break;
         }
     }
 
     return ret;
 }

 //beginがexpectedを指していたらbeginを一つ進める。
 void consume(State &begin, char expected) {
     if (*begin == expected) {
         begin++;
     } else {
         cerr << "Expected '" << expected << "' but got '" << *begin << "'" << endl;
         cerr << "Rest string is '";
         while (*begin) {
             cerr << *begin++;
         }
         cerr << "'" << endl;
         //throw ParseError();
     }
 }

snippet     power
abbr        冪乗
options     head
    
 //x^n mod M
 typedef unsigned long long ull;
 ull power(ull x, ull n, ull M){
     ull res = 1;
     if(n > 0){
         res = power(x, n / 2, M);
         if(n % 2 == 0) res = (res * res) % M;
         else res = (((res * res) % M) * x ) %M;
     }
     return res;

snippet     rotate
abbr        行列の回転
options     head
 //P is structure of coodinate.
 void rotationMatrix(P &p, double angle){
     double x, y;
         x = p.x * cos(angle) - p.y * sin(angle);
         y = p.x * sin(angle) + p.y * cos(angle);
         p.x = x;
         p.y = y;
 }

snippet     toNum
abbr        文字列から数値への変換
options     head
    
 //文字列から数値への変換
 int toNum(string str){
    int num = 0;
    rep(i,str.size()){
        num *= 10;
        num += str[i] - '0';
    }
    return num;
 }

snippet     toStr
abbr        数値から文字列への変換
options     head
    
 string toStr(int n){
    string str;
    int len = static_cast<int>(log10(n));
    int K = 1;
    rep(i,len) K*=10;
    rep(i,len + 1){
        if(n / K == 0) str+= '0';
        else str+= ('0' + n / K);
        n%=K; K/=10;
    }
    return str;
 }

snippet     union-find
abbr        森
options     head
    
 const int gmax_n = ;
 
 int par[gmax_n]; //親
 int depth[gmax_n];//木の深さ
 
 void init(int n){
     rep(i,n){
         par[i] = i;
         depth[i] = 0;
     }
 }
 
 int find(int x){
     if(par[x] == x){
         return x;
     }else {
         return par[x] = find(par[x]);
     }
 }
 
 void unite(int x, int y){
     x = find(x);
     y = find(y);
     if(x == y) return;
 
     if(depth[x] < depth[y]){
         par[x] = y;
     }else{
         par[y] = x;
         if(depth[x] == depth[y]) depth[x]++;
     }
 }
 
 bool same(int x, int y){
     return find(x) == find(y);
 }
 
snippet     warchallFloyd
abbr        全点最短経路
options     head

 const int MAX_V = ;

 void init(int m[MAX_V][MAX_V]){
     rep(i,MAX_V) rep(j,MAX_V) m[i][j] = INF;
     rep(i,MAX_V) m[i][i] = 0;
 }

 void warshallFloyd(int m[MAX_V][MAX_V], int n){
     rep(k,n){
         rep(i,n){
             rep(j,n){
                 m[i][j] = min(m[i][j], m[i][k] + m[k][j]);
             }
         }
     }
 }

snippet     dijkstraOfAdjacencyList
abbr        隣接リストのダイクストラ
options     head
    
 class Edge{
     public:
     int to, cost;
     Edge(int to, int cost) : to(to) ,cost(cost) {}
 };
 
 class Node{
     public:
     int dis;
     bool isUsed;
     Node(){
         this->dis = INF;
         this->isUsed = 0;
     }
 };
 
 typedef vector<vector<Edge>> AdjList;
 
 int dijkstra(AdjList g, int start, int n){
     vector<Node> node(n);
     priority_queue<int, vector<pair<int, int>>, greater<pair<int, int>>> q;
 
     q.push(make_pair(0, start));
     node[start].dis = 0;
 
     pair<int, int> u;
     while(not q.empty()){
         u = q.top(); q.pop();
         int current = u.second;
         node[current].isUsed = 1;
 
         rep(i,g[current].size()){
             int next = g[current][i].to;
             if(node[next].isUsed == 0){
                 if(node[next].dis > node[current].dis + g[current][i].cost){
                     node[next].dis = node[current].dis + g[current][i].cost;
                     q.push(make_pair(node[next].dis, next));
                 }
             }
         }
     }
     return 
 }

snippet     bellmanFord
abbr        負の経路を含む最短経路、負の経路の検出
options     head

 class Edge{
     public:
         int to, cost;
         Edge(int to, int cost) : to(to) ,cost(cost) {}
 };
 
 typedef vector<vector<Edge>> AdjList;
 vector<int> dis;
 
 bool bellmanFord(AdjList g, int n, int s) { // nは頂点数、sは開始頂点
     dis = vector<int>(n, INF);
     dis[s] = 0; // 開始点の距離は0
     rep(i,n){
         rep(v,n){
             rep(k,g[v].size()){
                 Edge e = g[v][k];
                 if (dis[v] != INF && dis[e.to] > dis[v] + e.cost) {
                     dis[e.to] = dis[v] + e.cost;
                     if (i == n - 1) return true; // n回目にも更新があるなら負の閉路が存在
                 }
             }
         }
     }
     return false;
 }
 
snippet     prim
abbr        最小全域木
options     head

 class Edge {
     public:
     int to, cost;
     Edge(int to, int cost) : to(to), cost(cost) { }
 };
 
 typedef vector<Edge> Edges;
 typedef vector<Edges> Graph;
 
 int prim(const Graph &g, int s = 0) {
     int n = g.size();
     int total = 0;
 
     vector<bool> visited(n);
     //priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
     priority_queue<pair<int, int> > q;
 
     q.push(make_pair(0,s));
     while (not q.empty()) {
         pair<int, int> u = q.top(); q.pop();
         if (visited[u.second]) continue;
         total += u.first;
         visited[u.second] = true;
         for(auto it : g[u.second]) {
             if (not visited[it.to]) q.push(make_pair(it.cost, it.to));
         }
     }
     return total;
 }

snippet     binaryIndexedTree
abbr        セグメント木 合計
options     head
 
 const int MAX_N = 100005;
 
 //[1, n]
 int bit[MAX_N + 1] = {0};
 
 int sum(int i){
     int s = 0;
     while(i > 0){
         s += bit[i];
         i -= i & -i;
     }
     return s;
 }
 
 void add(int i, int x, int n){
     while(i <= MAX_N){
         //bit[i] += x;
         bit[i] = max(bit[i], x);
         i += i & - i;
     }
 }

snippet     LCS
abbr        最長共通部分文字列の長さ
options     head

 const int MAX_N = 1000;
 int dp[MAX_N + 1][MAX_N + 1];
 int n, m; //文字数、共通部分文字列の長さ
 string s, t;
 
 int solve(){
     rep(i,n){
         rep(j,m){
             if(s[i] == t[j]){
                 dp[i + 1][j + 1] = dp[i][j] + 1;
             }else{
                 dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j]);
             }
         }
     }
     return dp[n][m];
 }

snippet     LIS
abbr        最長増加部分列
options     head
 const int MAX_N = 1000;
 int dp[MAX_N];

 int LIS(int n, int a[MAX_N]){
     fill(dp, dp + n, INF);
     rep(i,n){
         *lower_bound(dp, dp + n, a[i]) = a[i];
     }
     return lower_bound(dp, dp + n, INF) - dp;
 }

snippet     divisor
abbr        約数の列挙
options     head
 vector<int> divisor(int n){
     vector<int> res;
     for(int i = 1; i * i <= n; i++){
         if(n % i == 0){
             res.emplace_back(i);
             if(i != n / i) res.emplace_back(n / i);
         }
     }
     return res;
 }

snippet     primeVactor
abbr        素因数分解
options     head
 map<int, int> prime_factor(int n){
     map<int, int> res;
     for(int i = 2; i * i <= n; i++){
         while(n % i == 0){
             ++res[i];
             n /= i;
         }
     }
     if(n != 1) res[n] = 1;
     return res;
 }

snippet     syakutori
abbr        しゃくとり
options     head
 //S:sumの条件 n:個数
 int solve(){
     int res = n + 1;
     int s = 0, t = 0, sum = 0;
     while(true){
         while(t < n && sum < S){
             sum += a[t++];
         }
         if(sum < S) break;
         res = min(res, t - s);
         sum -= a[s++];
     }
     if(res > n){
         res = 0;
     }
     return res;
 }

snippet     isUniqueStr
abbr        文字列の文字がユニークか判定
options     head
    
 bool isUniqueStr(string s){
     unsigned int char_set[128] = {0};
     rep(i,s.size()) char_set[s[i]]++;
     rep(i,128) if(char_set[i] >= 2) return 0;
     return 1;
 }

snippet     areAnagram
abbr        文字列がアナグラムかを判定
options     head

 bool areAnagram(string s, string t){
     if(s.size() != t.size()) return 0;
 
     unsigned int char_set[128] = {0};
     rep(i,s.size()){
         char_set[s[i]]++;
         char_set[t[i]]--;
     }
     rep(i,128) if(char_set[i] != 0) return 0;
     return 1;
 }

snippet     split
abbr        文字列を空白で区切る
options     head

 vector<string> split(string in){
     vector<string> ret;
     stringstream str(in);
     string s;
     while(str >> s){
         ret.emplace_back(s);
     }
     return ret;
 }

snippet     articulationPoints
abbr        無向グラフの関節点を全列挙
options     head

 const int MAX_V = 100000;
 
 class Node{
     public:
         bool isUsed;
         int prenum; //DFSの訪問の順番
         int parent; //DFS Treeにおける親
         int lowest; //min(自分のprenum, 逆辺がある場合の親のprenum, すべての子のlowest)
         Node(){ this->isUsed = 0; }
 };
 
 int cnt = 1;
 vector<Node> node(MAX_V);
 vector<int> edge[MAX_V];
 vector<pair<int, int>> bridge;
 
 void dfs(int current, int prev){
     node[current].prenum = node[current].lowest = cnt;
     cnt++;
 
     node[current].isUsed = true;
 
     int next;
     rep(i,edge[current].size()){
         next = edge[current][i];
         if(not node[next].isUsed){
             node[next].parent = current;
             dfs(next, current);
             node[current].lowest = min(node[current].lowest, node[next].lowest);
             if(node[current].prenum < node[next].lowest){
                 bridge.emplace_back(make_pair(min(current, next), max(current, next)));
             }
         }else if(next != prev){
             node[current].lowest = min(node[current].lowest, node[next].prenum);
         }
     }
 }
 
 void art_points(int v){
     dfs(0, -1); // 0 == root
 
     set<int> ap;
     int np = 0;
     range(i,1,v){
         int p = node[i].parent;
         if(p == 0) np++;
         else if(node[p].prenum <= node[i].lowest) ap.insert(p);
     }
     if(np > 1) ap.insert(0);
     //for(auto it:ap){ cout << it << endl; } //関節点の全列挙
     sort(all(bridge));
     for(auto it:bridge){ cout << it.first << ' ' << it.second << endl; } //橋の全列挙
 
 }

snippet     stronglyConnectedComponents
abbr        強連結性成分分解
options     head

 const int MAX_V = 10000;
 
 vector<int> g[MAX_V]; //グラフ
 vector<int> rg[MAX_V]; //辺が逆になったグラフ
 vector<int> vs; //帰りがけ順の並び
 bool used[MAX_V];
 int cmp[MAX_V]; //属する強連結成分のトポロジカル順序
 
 void addEdge(int from, int to){
     g[from].emplace_back(to);
     rg[to].emplace_back(from);
 }
 
 void dfs(int v){
     used[v] = true;
     rep(i,g[v].size()){
         if(not used[g[v][i]]) dfs(g[v][i]);
     }
     vs.emplace_back(v);
 }
 
 void rdfs(int v, int k){
     used[v] = true;
     cmp[v] = k;
     rep(i,rg[v].size()){
         if(not used[rg[v][i]]) rdfs(rg[v][i], k);
     }
 }
 
 int scc(int v){
     memset(used, 0, sizeof(used));
     vs.clear();
     rep(i,v){
         if(not used[i]) dfs(i);
     }
     memset(used, 0, sizeof(used));
     int k = 0;
     for(int i = vs.size() - 1; i >= 0; i--){
         if(not used[vs[i]]) rdfs(vs[i], k++);
     }
     return k;
 }

snippet     topologicalSort
abbr        トポロジカルソート
options     head

 const int MAX_V = 10000;
 
 vector<int> g[MAX_V]; //グラフ
 vector<int> tp; //トポロジカルソートの結果
 
 void bfs(int s, int indeg[MAX_V], bool used[MAX_V]){
     queue<int> q;
 
     q.push(s);
     used[s] = true;
 
     while(not q.empty()){
         int u = q.front(); q.pop();
         tp.emplace_back(u);
         rep(i,g[u].size()){
             int v = g[u][i];
             indeg[v]--;
             if(indeg[v] == 0 && not used[v]){
                 used[v] = true;
                 q.push(v);
             }
         }
     }
 }
 
 //グラフに閉路がある場合、0を返す
 bool topologicalSort(int v){
     int indeg[MAX_V]; //入次数
     bool used[MAX_V];
     memset(indeg, 0, sizeof(indeg));
     memset(used, 0, sizeof(used));
 
     rep(i,v) rep(j,g[i].size()) indeg[ g[i][j] ]++;
     rep(i,v) if(indeg[i] == 0 && not used[i]) bfs(i, indeg, used);
 
     for(auto it:tp) cout << it << endl;
 
     if(tp.size() == v) return true;
     else return false;
 }

snippet     diameterOfTree
abbr        木の直径を求める
options     head

 const int MAX_V = 100000;
 
 class Edge{
     public:
     int to, dis;
     Edge(){}
     Edge(int to, int dis): to(to), dis(dis)  {}
 };
 
 vector<Edge> g[MAX_V];
 int dis[MAX_V];
 
 void bfs(int s, int n){
     queue<int> q;
 
     rep(i,n) dis[i] = INF;
     dis[s] = 0;
     q.push(s);
 
     int u;
     while(not q.empty()){
         u = q.front(); q.pop();
         rep(i,g[u].size()){
             Edge e = g[u][i];
             if(dis[e.to] == INF){
                 dis[e.to] = dis[u] + e.dis;
                 q.push(e.to);
             }
         }
     }
 }
 
 void solve(int n){
     int maxi = 0;
     int tgt = 0; //ある点sから最も遠い点
 
     bfs(0, n);
     rep(i,n){
         if(dis[i] == INF) continue;
         if(maxi < dis[i]){
             maxi  = dis[i];
             tgt = i;
         }
     }
 
     bfs(tgt, n);
     maxi = 0; //tgtから最も遠い接点の距離maxi
     rep(i,n){
         if(dis[i] == INF) continue;
         maxi = max(maxi, dis[i]);
     }
 
     cout << maxi << endl;
 }

snippet     heightOfTree
abbr        木の高さを求める
options     head
    
 const int MAX_V = 10000;
 
 class Edge{
     public:
         int dst, weight;
         Edge(){}
         Edge(int dst, int weight): dst(dst), weight(weight)  {}
 };
 
 typedef vector<vector<Edge>> Graph;
 
 Graph g(MAX_V);
 
 int visit(Graph &t, int i, int j) {
     if(t[i][j].weight >= 0) return t[i][j].weight;
     t[i][j].weight = g[i][j].weight;
     int u = t[i][j].dst;
     rep(k,t[u].size()) {
         if(t[u][k].dst == i) continue;
         t[i][j].weight = max(t[i][j].weight, visit(t,u,k) + g[i][j].weight);
     }
     return t[i][j].weight;
 }
 vector<int> height(int n) {
     Graph t = g;
     rep(i,n) rep(j,t[i].size()) t[i][j].weight = -1;
     rep(i,n) rep(j,t[i].size()) if(t[i][j].weight < 0) t[i][j].weight = visit(t, i, j);
 
     vector<int> ht(n); // gather results
     rep(i,n) rep(j,t[i].size()) ht[i] = max(ht[i], t[i][j].weight);
     return ht;
 }


snippet     minCostFlow
abbr        最小費用流
options     head
    
 const int MAX_V = 101;
 
 class Edge{
     public:
         //行き先、容量、コスト、逆辺
         int to, cap, cost, rev;
         Edge(int to, int cap, int cost, int rev) to(to), cap(cap), cost(cost), rev(rev){}
 };
 
 vector<vector<Edge>> G(MAX_V);
 int h[MAX_V]; //ポテンシャル
 int dist[MAX_V]; //最短距離
 int prev_v[MAX_V], prev_e[MAX_V]; //直前の頂点と辺
 
 void addEdge(int from, int to, int cap, int cost){
     G[from].emplace_back(Edge(to, cap, cost, static_cast<int>(G[to].size())));
     G[to].emplace_back(Edge(from, 0, -cost, static_cast<int>(G[from].size() - 1)));
 }
 
 int minCostFlow(int v, int s, int t, int f){
     int res = 0;
     fill(h, h + v, 0);
     while(f > 0){
         priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
         fill(dist, dist + v, INF);
         dist[s] = 0;
         q.push(make_pair(0, s));
         while(not q.empty()){
             pair<int, int> p = q.top(); q.pop();
             int u = p.second;
             if(dist[u] < p.first) continue;
             rep(i,G[u].size()){
                 Edge &e = G[u][i];
                 if(e.cap > 0 && dist[e.to] > dist[u] + e.cost + h[u] - h[e.to]){
                     dist[e.to] = dist[u] + e.cost + h[u] - h[e.to];
                     prev_v[e.to] = u;
                     prev_e[e.to] = i;
                     q.push(make_pair(dist[e.to], e.to));
                 }
             }
         }
         if(dist[t] == INF){
             return -1;
         }
         rep(i,v) h[i] += dist[i];
 
         int d = f;
         for(int u = t; u != s; u = prev_v[u]){
             d = min(d, G[prev_v[u]][prev_e[u]].cap);
         }
         f -= d;
         res += d * h[t];
         for(int u = t; u != s; u = prev_v[u]){
             Edge &e = G[prev_v[u]][prev_e[u]];
             e.cap -= d;
             G[u][e.rev].cap += d;
         }
     }
     return res;
 }

snippet     bipartiteMatching
abbr        2部マッチング
options     head

 const int MAX_V = 210; //MAX_X + MAX_Y
 const int MAX_X = 105;
 const int MAX_Y = 105;
 
 class Edge{
     public:
         int to, cap, rev;
 };
 
 typedef vector<vector<Edge>> AdjList;
 AdjList G(MAX_V);
 bool used[MAX_V];
 
 void addEdge(int from, int to, int cap){
     G[from].emplace_back(Edge{to, cap, static_cast<int>(G[to].size())});
     G[to].emplace_back(Edge{from, 0, static_cast<int>(G[from].size() - 1)});
 }
 
 int dfs(int v, int t, int f){
     if(v == t) return f;
     used[v] = true;
     rep(i,G[v].size()){
         Edge &e = G[v][i];
         if(not used[e.to] && e.cap > 0){
             int d = dfs(e.to, t, min(f, e.cap));
             if(d > 0){
                 e.cap -= d;
                 G[e.to][e.rev].cap += d;
                 return d;
             }
         }
     }
     return 0;
 }
 
 int maxFlow(int s, int t){
     int flow = 0;
     while(true){
         memset(used, 0, sizeof(used));
         int f = dfs(s, t, INF);
         if(f == 0) return flow;
         flow += f;
     }
 }
 
 int bipartiteMatching(int x, int y, bool edge[MAX_X][MAX_Y]){
     int s = x + y, t = s + 1; //set x : 0 ~ x-1, set y : x ~ x+y-1
 
     rep(i,x) addEdge(s, i, 1); //sと集合xを結ぶ
     rep(i,y) addEdge(x + i, t, 1); //集合yとtを結ぶ
 
     rep(i,x) rep(j,y) if(edge[i][j]) addEdge(i, x + j, 1); //集合xと集合yを結ぶ
 
     return maxFlow(s, t);
 }

snippet     levenshteinDistance
abbr        編集距離を求めるdp
 const int MAX_N = 1005;
 int dp[MAX_N][MAX_N];
 
 int minimum(int a, int b, int c){
     return min(min(a,b),c);
 }
 
 int levenshteinDistance(string a, string b){
     rep(i,a.size() + 1) dp[i][0] = i;
     rep(i,b.size() + 1) dp[0][i] = i;
 
     range(i,1,a.size() + 1){
         range(j,1,b.size() + 1){
             int cost = a[i - 1] == b[j - 1] ? 0 : 1;
             dp[i][j] = minimum(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
         }
     }
     return dp[a.size()][b.size()];
 }

snippet     cumulativeSum
abbr        2次元累積和
 const int MAX_N = ;
 const int MAX_P = MAX_N * MAX_N;

 int g[MAX_N][MAXN];
 
 //s[MAX_N + 1][MAX_N + 1]
 void cumulativeSum(int h, int w){
     //左上の要素は(1,1)
     rep(i,h) rep(j,w) g[i + 1][j + 1] += g[i + 1][j] + g[i][j + 1] + g[i][j];
 }
 
 //(i,j)を左上、(k,l)を右下とした長方形
 int sum(int lx, int ly, int rx, int ry){
     return s[ry][rx] - s[ry][lx] - s[ly][rx] + s[ly][lx];
 }
 
 //全ての大きさ、場所の和を求めるループ。
 int fullSearch(int maxi[MAX_P], int n){
     range(i,1,n + 1){
         range(j,1,n + 1){
             range(k,i,n + 1){
                 range(l,j,n + 1){
                     int x = (k - i + 1) * (l - j + 1);
                     maxi[x] = max(maxi[x], sum(i,j,k,l));
                 }
             }
         }
     }
 }

snippet     oneStrokePath
abbr        全ての頂点を通る一筆書きの総数
 const int MAX_V = 10;
 
 int n;
 int ans = 0;
 bool M[MAX_V][MAX_V] = {0};
 
 void dfs(int c, int u, vector<bool> used){
     if(u == n - 1){
         ans++;
         return;
     }
     rep(i,n){
         if(M[c][i] == 1 && used[i] == false){
             used[c] = true;
             dfs(i,u + 1,used);
         }
     }
 }

snippet     bitManip
abbr        ビットの操作
 //i番目のビットを返す
 bool getBit(int num, int i){
     return ((num & (1 << i)) != 0);
 }
 
 //i番目を1にする
 int setBit(int num, int i){
     return num | (1 << i);
 }
 
 //i番目を0にする
 int clearBit(int num, int i){
     int mask = ~(1 << i);
     return num & mask;
 }
 
 //i番目をvで置き換える
 int updateBit(int num, int i, int v){
     int mask = ~(1 << i);
     return (num & mask) | (v << i);
 }

snippet     numPrimeFactor
abbr        約数の個数を求める
 const long long M = 1000000007;
 map<int, int> res;
 
 void primeFactor(int n){
     for(int i = 2; i * i <= n; i++){
         while(n % i == 0){
             ++res[i];
             n /= i;
         }
     }
     if(n != 1) res[n] += 1;
 }
 
 long long numPrimeFactor(){
     long long ans = 1;
     for(auto it:res){
         ans = ans * (it.second + 1);
         ans %= M;
     }
     return ans;
 }

snippet     eightQueensProblem
abbr        8クイーン問題
 const int N = 8;
 
 //row[i] = j : (i,j)にクイーン
 int row[N], col[N], dpos[2 * N - 1], dneg[2 * N - 1];
 char x[8][8];
 bool f = false;
 
 void init(){
     rep(i,N){
         row[i] = 0;
         col[i] = 0;
     }
     rep(i,2 * N - 1){
         dpos[i] = 0;
         dneg[i] = 0;
     }
 }
 
 void printBoard(){
     rep(i,N){
         rep(j,N){
             if(x[i][j] == 'Q'){
                 if(row[i] != j) return;
             }
         }
     }
     rep(i,N){
         rep(j,N){
             cout << ( (row[i] == j) ? "Q" : ".");
         }
         cout << endl;
     }
     f = true;
 }
 
 void recursive(int i){
     if(i == N){
         printBoard();
         return;
     }
 
     rep(j,N){
         if(col[j] || dpos[i + j] || dneg[i - j + N - 1]) continue;
         row[i] = j;
         col[j] = dpos[i + j] = dneg[i - j + N - 1] = 1;
         recursive(i + 1);
         row[i] = col[j] = dpos[i + j] = dneg[i - j + N - 1] = 0;
         if(f) return;
     }
 }

snippet     next_combination
abbr        組み合わせの全列挙
 template < class BidirectionalIterator >
 bool next_combination ( BidirectionalIterator first1 ,
         BidirectionalIterator last1 ,
         BidirectionalIterator first2 ,
         BidirectionalIterator last2 ){
     if (( first1 == last1 ) || ( first2 == last2 )) {
         return false ;
     }
     BidirectionalIterator m1 = last1 ;
     BidirectionalIterator m2 = last2 ; --m2;
     while (--m1 != first1 && !(* m1 < *m2 )){
     }
     bool result = (m1 == first1 ) && !(* first1 < *m2 );
     if (! result ) {
         while ( first2 != m2 && !(* m1 < * first2 )) {
             ++ first2 ;
         }
         first1 = m1;
         std :: iter_swap (first1 , first2 );
         ++ first1 ;
         ++ first2 ;
     }
     if (( first1 != last1 ) && ( first2 != last2 )) {
         m1 = last1 ; m2 = first2 ;
         while (( m1 != first1 ) && (m2 != last2 )) {
             std :: iter_swap (--m1 , m2 );
             ++ m2;
         }
         std :: reverse (first1 , m1 );
         std :: reverse (first1 , last1 );
         std :: reverse (m2 , last2 );
         std :: reverse (first2 , last2 );
     }
     return ! result ;
 }
 
 template < class BidirectionalIterator > bool next_combination ( BidirectionalIterator first , BidirectionalIterator middle , BidirectionalIterator last )
 {
     return next_combination (first , middle , middle , last );
 }
 
 //要素vからr個取り出す組み合わせ
 void func(vector<int> v, int r){
     do{
     }while(next_combination(v.begin(), v.begin() + r, v.end()));
 }

snippet     numOfnumber
abbr        nまでの数字を書いたとき、1が出現する回数
 //nに含まれる1の数
 long long numOfnumber(long long n){
     long long k = 10;
     long long ans = 0;
     rep(i,9){
         ans += n / k * (k / 10LL);
         ans += min(max(0LL, n % k - (k / 10LL - 1)), k / 10LL);
         k*=10LL;
     }
     return ans;
 }

snippet     ternarySearch
abbr        三分探索
 double C(double x){
 }
 
 double ternarySearch(double p){
     double right = INF, left = 0;
     rep(i,200){
         double llr = (left * 2 + right) / 3;
         double rll = (left + right * 2) / 3;
         if(C(llr) > C(rll)){
             left = llr;
         }else{
             right = rll;
         }
     }
     return left;
 }

snippet     combination
abbr        べき乗/階乗/コンビネーション
 typedef unsigned long long ull;
 const ull M = 1000000007;
 
 //べき乗 x^n mod M
 ull power(ull x, ull n){
     ull res = 1;
     if(n > 0){
         res = power(x, n / 2);
         if(n % 2 == 0) res = (res * res) % M;
         else res = (((res * res) % M) * x ) % M;
     }
     return res;
 }
 
 //階乗
 ull factorial(int n){
     ull res = 1;
     range(i,1,n + 1){
         res*= i;
         res%= M;
     }
     return res;
 }
 
 //nCr コンビネーション (1,1)から(w,h)だと、引数は(w - 1, h - 1, M)
 ull combination(ull n, ull r){
     //nCr = n! / ( (n - r)! * r! )
     ull a = factorial(n);
     ull b = factorial(n - r) * factorial(r) % M;
     return a * power(b, M - 2) % M;
 }

snippet     toDAG
abbr        2点間の最短経路になる辺を残したDAG
 //出発地、出発地から全ての点へ対する最短経路、返り値、辺
 void toDAG(int s, int g[MAX_V][MAX_V], int dag[MAX_V][MAX_V], vector<pair<int, int>> v){
     rep(i,v.size()){
         if(g[s][v[i].first] + 1 == g[s][v[i].second]){
             dag[v[i].first][v[i].second] = 1;
         }
     }
 }

snippet     intervalState
abbr        区間の関係
 //区間A[a,b]と区間B[c,d]の関係
 int intervalState(int a, int b, int c, int d){
     if(a < c && b < c) return 0;            //A < B
     else if(a > d && b > d) return 1;       //A > B
     else if(a <= c && d <= b) return 2;     //A -> B
     else if(c < a && b < d) return 3;       //B -> A
     else if(a <= c && b < d) return 4;      //A <= B
     else if(c < a && d <= b) return 5;      //A >= B
     return -1;
 }

snippet     pascals_triangle
abbr        n個を選ぶ組み合わせの中、k個を選ぶ組み合わせの割合。
 //n個を選ぶ組み合わせの中、k個を選ぶ組み合わせの割合。
 void Pascals(double m[N][N]){
     m[0][0] = 1;
     range(i,1,1011){
         rep(j,i + 1){
             if(j == 0) m[i][j] = m[i - 1][j] / 2;
             else if(j == i) m[i][j] = m[i - 1][j - 1] / 2;
             else m[i][j] = (m[i - 1][j] + m[i - 1][j - 1]) / 2;
         }
     }
 }

snippet     dp_Partial_sum_with_number_restriction
abbr        個数制限付き部分和

 const int MAX_N = 105;
 const int MAX_K = 100005;

 void solve(){
     int n, k;
 
     scanf("%d%d", &n, &k);
     int a[MAX_N], m[MAX_N];
     rep(i,n) scanf("%d", &a[i]);
     rep(i,n) scanf("%d", &m[i]);
 
     int dp[MAX_K];
     memset(dp, -1, sizeof(dp));
     dp[0] = 0;
     rep(i,n){
         rep(j,k + 1){
             if(dp[j] >= 0){
                 dp[j] = m[i];
             }else if(j < a[i] || dp[j - a[i]] <= 0){
                 dp[j] = -1;
             }else{
                 dp[j] = dp[j - a[i]] - 1;
             }
         }
     }
 
     int sum = 0;
     range(i,1,k + 1){
         if(dp[i] >= 0) sum++;
     }
     cout << sum << endl;
 }

snippet     gridUnion-find
abbr        グリッドグラフのユニオン木
 const int gmax_n =1005 ;
 const int dy[3] = { 0, 1, 1};
 const int dx[3] = { 1, 0, 1};
 
 pair<int, int> par[gmax_n][gmax_n]; //親
 int depth[gmax_n][gmax_n];//木の深さ
 int cc[gmax_n][gmax_n]; //連結成分
 
 void init(int h, int w){
     rep(i,h){
         rep(j,w){
             par[i][j] = make_pair(i,j);
             depth[i][j] = 0;
             cc[i][j] = 1;
         }
     }
 }
 
 pair<int, int> find(pair<int, int> x){
     if(par[x.first][x.second] == x){
         return x;
     }else {
         return par[x.first][x.second] = find(par[x.first][x.second]);
     }
 }
 
 void unite(pair<int, int> x, pair<int, int> y){
     x = find(x);
     y = find(y);
     if(x == y) return;
 
     if(depth[x.first][x.second] < depth[y.first][y.second]){
         par[x.first][x.second] = y;
         cc[y.first][y.second] += cc[x.first][x.second];
     }else{
         par[y.first][y.second] = x;
         cc[x.first][x.second] += cc[y.first][y.second];
         if(depth[x.first][x.second] == depth[y.first][y.second]) depth[x.first][x.second]++;
     }
 }
 
 bool same(pair<int, int> x, pair<int, int> y){
     return find(x) == find(y);
 }
 
 void uniteAll(int h, int w, char m[gmax_n][gmax_n]){
     rep(i,h){
         rep(j,w){
             if(m[i][j] == 'o'){
                 rep(k,3){
                     int ny = i + dy[k];
                     int nx = j + dx[k];
                     if(ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
                     if(m[ny][nx] == 'o') unite(make_pair(i,j), make_pair(ny,nx));
                 }
             }
             if(i < h && j < w && m[i + 1][j] == 'o' && m[i][j + 1] == 'o'){
                 unite(make_pair(i + 1, j), make_pair(i, j + 1));
             }
         }
     }
 }
 
 void check(int n, int ans[3]){
     int i = 0;
     while(true){
         i++;
         assert(i <= 310);
         if(n % (i * i) != 0) continue;
 
         if(n / (i * i) == 12){ ans[0]++; break; }
         else if(n / (i * i) == 16){ ans[1]++; break; }
         else if(n / (i * i) == 11){ ans[2]++; break; }
     }
 }
 
 void print(int h, int w, char m[gmax_n][gmax_n]){
     rep(i,h){ rep(j,w){ cout << cc[i][j] << ' '; } cout << endl; } cout << endl;
     rep(i,h){ rep(j,w){
         if(make_pair(i,j) == par[i][j]) cout << '.';
         else cout << '#';
     } cout << endl; } cout << endl;
 }

snippet     lowest common ancestor - doubling
abbr        ダブリングを利用したLCA
 const int MAX_V = 100005;
 const int MAX_LOG_V = 20;
 
 vector<int> g[MAX_V];
 int root;
 
 int parent[MAX_LOG_V][MAX_V];
 int depth[MAX_V];
 
 void dfs(int v, int p, int d){
     parent[0][v] = p;
     depth[v] = d;
     rep(i,g[v].size()){
         if(g[v][i] != p) dfs(g[v][i], v, d + 1);
     }
 }
 
 void init(int V){
     root = 0; //ココ
     dfs(root, -1, 0);
     rep(k,MAX_LOG_V - 1){
         rep(v,V){
             if(parent[k][v] < 0) parent[k + 1][v] = -1;
             else parent[k + 1][v] = parent[k][ parent[k][v] ];
         }
     }
 }
 
 int lca(int u, int v){
     if(depth[u] > depth[v]) swap(u, v);
     rep(k,MAX_LOG_V){
         if( (depth[v] - depth[u]) >> k & 1){
             v = parent[k][v];
         }
     }
     if(u == v) return u;
     for(int k = MAX_LOG_V - 1; k >= 0; k--){
         if(parent[k][u] != parent[k][v]){
             u = parent[k][u];
             v = parent[k][v];
         }
     }
     return parent[0][u];
 }

snippet     compress coordinate
abbr        座標圧縮
 void compress(vector<int> &v) {
     sort(v.begin(), v.end());
     v.erase(unique(v.begin(),v.end()),v.end());
 }
 
 int lb(vector<int> v, int num){
     return lower_bound(all(v), num) - v.begin();
 }

snippet     meet_in_the_middle
abbr        bitによる全通列挙
 //要素wをnとmに分け、それぞれで全列挙する
 vector<long long> a, b;
 rep(i,(1 << n)){
     long long sum = 0;
     rep(j,n){
         if(getBit(i,j)) sum += w[j];
     }
     a.emplace_back(sum);
 }
 rep(i,(1 << m)){
     long long sum = 0;
     rep(j,m){
         if(getBit(i,j)) sum += w[n + j];
     }
     b.emplace_back(sum);
 }

snippet     matrix
abbr        行列計算
 const int M = 10000;
 typedef vector<vector<int>> mat;
 
 mat mul(mat &a, mat &b){
     mat c(a.size(), vector<int>(b[0].size()));
     rep(i,a.size()){
         rep(k,b.size()){
             rep(j,b[0].size()){
                 c[i][j] = (c[i][j] + a[i][k] * b[k][j]) % M;
             }
         }
     }
     return c;
 }
 
 mat pow(mat a, int n){
     mat b(a.size(), vector<int>(a.size()));
     rep(i,a.size()){
         b[i][i] = 1;
     }
     while(n > 0){
         if(n & 1) b = mul(b,a);
         a = mul(a, a);
         n >>= 1;
     }
     return b;
 }
 
 int solve(int n){
     mat a(2, vector<int>(2));
     a[0][0] = 1; a[0][1] = 1;//フィボナッチ数列の漸化式の行列
     a[1][0] = 1; a[1][1] = 0;
     a = pow(a,n); //行列Aのn乗。
     return a[1][0];
 }

snippet     flow
abbr        最大流
 const int MAX_V = 10005;
 
 class Edge{
     public:
         int to, cap, rev;
         Edge(int to, int cap, int rev) : to(to), cap(cap), rev(rev) {}
 };
 
 class Flow{
     private:
         vector<Edge> G[MAX_V];
         bool used[MAX_V];
         int level[MAX_V]; //sからの距離
         int iter[MAX_V]; //どこまで調べ終わったか
         int dfs(int v, int t, int f){
             if(v == t) return f;
             used[v] = true;
             rep(i,G[v].size()){
                 Edge &e = G[v][i];
                 if(not used[e.to] && e.cap > 0){
                     int d = dfs(e.to, t, min(f, e.cap));
                     if(d > 0){
                         e.cap -= d;
                         G[e.to][e.rev].cap += d;
                         return d;
                     }
                 }
             }
             return 0;
         }
         int dfs_(int v, int t, int f){
             if(v == t) return f;
             for(int &i = iter[v]; i < G[v].size(); i++){
                 Edge &e = G[v][i];
                 if(e.cap > 0 && level[v] < level[e.to]){
                     int d = dfs_(e.to, t, min(f, e.cap));
                     if(d > 0){
                         e.cap -= d;
                         G[e.to][e.rev].cap += d;
                         return d;
                     }
                 }
             }
             return 0;
         }
         void bfs(int s){
             memset(level, -1, sizeof(level));
             queue<int> que;
             level[s] = 0;
             que.push(s);
             while(not que.empty()){
                 int v = que.front(); que.pop();
                 rep(i,G[v].size()){
                     Edge &e = G[v][i];
                     if(e.cap > 0 && level[e.to] < 0){
                         level[e.to] = level[v] + 1;
                         que.push(e.to);
                     }
                 }
             }
         }
     public:
         void addEdge(int from, int to, int cap){
             G[from].push_back(Edge(to, cap, static_cast<int>(G[to].size())));
             G[to].push_back(Edge(from, 0, static_cast<int>(G[from].size() - 1)));
         }
         int fordFulkerson(int s, int t){
             int flow = 0;
             while(true){
                 memset(used, 0, sizeof(used));
                 int f = dfs(s, t, INF);
                 if(f == 0) return flow;
                 flow += f;
             }
         }
         int dinic(int s, int t){
             int flow = 0;
             while(true){
                 bfs(s);
                 if(level[t] < 0) return flow;
                 memset(iter, 0, sizeof(iter));
                 int f;
                 while( (f = dfs_(s, t, INF)) > 0){
                     flow += f;
                 }
             }
         }
 };

snippet     simultaneousLinearEquations
abbr        連立一次方程式
 const double EPS = 1e-8;
 typedef vector<double> vd;
 typedef vector<vd> mat;
 
 vd simultaneousLinearEquations(const mat &A, const vd &b){
     int n = A.size();
     mat B(n, vd(n + 1));
     rep(i,n) rep(j,n) B[i][j] = A[i][j];
     rep(i,n) B[i][n] = b[i];
 
     rep(i,n){
         int pivot = i;
         for(int j = i; j < n; j++){
             if(abs(B[i][j]) > abs(B[pivot][i])) pivot = j;
         }
         swap(B[i], B[pivot]);
 
         if(abs(B[i][i]) < EPS) return vd(); //解なし or 一意ではない
 
         for(int j = i + 1; j <= n; j++) B[i][j] /= B[i][i];
         rep(j,n){
             if(i != j){
                 for(int k = i + 1; k <= n; k++) B[j][k] -= B[j][i] * B[i][k];
             }
         }
     }
     vd x(n);
     rep(i,n) x[i] = B[i][n];
     return x;
 }

snippet     extgcd
abbr        拡張ユークリッドの互除法
 //ax + by = gcd(a,b) の解をもとめる
 int extgcd(int a, int b, int &x, int &y){
     int d = a;
     if(b != 0){
         d = extgcd(b, a % b, y, x);
         y -= (a / b) * x;
     }else{
         x = 1; y = 0;
     }
     return d; //gcd(x,y)
 }

snippet     eulerPhi
abbr        オイラー関数
 const int MAX_N = 100;
 
 //オイラー関数の値を求める
 int eulerPhi(int n){
     int res = n;
     for(int i = 2; i * i <= n; i++){
         if(n % i == 0){
             res = res / i * (i - 1);
             for(; n % i == 0; n /= i);
         }
     }
     if(n != 1) res = res / n * (n - 1);
     return res;
 }
 
 int euler[MAX_N];
 
 //オイラー関数のテーブルを作る
 void eulerPhi2(){
     rep(i,MAX_N) euler[i] = i;
     for(int i = 2; i < MAX_N; i++){
         if(euler[i] == i){
             for(int j = i; j < MAX_N; j += i) euler[j] = euler[j] / i * (i - 1);
         }
     }
 }

snippet     segTree
abbr        セグメントツリークラス
 const int MAX_N = 100010;
 
 class segTree{
     private:
         //セグメントツリーを持つ配列
         int n, dat[4 * MAX_N];
         int query(int a, int b, int k, int l, int r){
             //[a, b) と[l, r)が交差しなければ、INT_MAX
             if(r <= a || b <= l) return INT_MAX;
 
             //[a,b)が[l,r)を完全に含んでいれば、この節点の値
             if(a <= l && r <= b) return dat[k];
             else{
                 //そうでなければ、２つのこの最小値
                 int vl = query(a, b, k * 2, l, ( l + r) / 2);
                 int vr = query(a, b, k * 2 + 1, (l + r) / 2, r);
                 return min(vl, vr);
             }
         }
     public:
         void init(int n_){
             n = 1;
             while(n < n_) n *= 2;
             rep(i,2 * n) dat[i] = INT_MAX;
         }
         void init(int a[MAX_N], int n_){ //配列aでの初期化
             n = 1;
             while(n < n_) n *= 2;
             for(int i = n; i < n * n; i++){
                 dat[i] = a[i - n];
             }
             for(int i = n - 1; i >= 1; i--){
                 dat[i] = min(dat[i * 2], dat[i * 2 + 1]);
             }
         }
         void update(int i, int x){
             i += n; //葉の節点
             dat[i] = x;
             while(i > 0){ //登りながら更新
                 dat[i / 2] = min(dat[i], dat[i^1]);
                 i = i / 2;
             }
         }
         //[a, b)の最小値を求める
         int query(int a, int b){
             return query(a,b,1,0,n);
         }
 };

snippet     repalceAll
abbr        文字列の置き換え
 string replaceAll(string s, string from, string to){
     vector<int> all;
     string tmp = s, tmp_space = s;
 
     string::size_type pos = tmp.find(from);
     while(pos != string::npos){
         all.emplace_back(pos);
         pos = tmp.find(from, pos + from.size());
     }
 
     //string space(from.size(), ' ');
     rep(j,all.size()){
         tmp.replace(all[j] + (to.size() - from.size()) * j, from.size(), to);
         //tmp_space.replace(all[j] + (from.size() - to.size()) * j, from.size(), space);
     }
     //if(tmp_space.find(to) != string::npos) return "0";
     if(s == tmp || all.empty()) "0";
 
     return tmp;
 }

snippet     JoinInterval
abbr        区間の結合
 //区間A[a,b]と区間B[c,d]の関係
 int intervalState(int a, int b, int c, int d){
     if(a < c && b < c) return 0;            //A < B
     else if(a > d && b > d) return 1;       //A > B
     else if(a <= c && d <= b) return 2;     //A -> B
     else if(c < a && b < d) return 3;       //B -> A
     else if(a <= c && b < d) return 4;      //A <= B
     else if(c < a && d <= b) return 5;      //A >= B
     return -1;
 }
 
 //Give input directly to vector<pair<int, int>> in
 vector<pair<int, int>> JoinInterval(vector<pair<int,int>> in){
     vector<pair<int, int>> v;
     rep(i,in.size()) in[i].second *= -1;
     sort(all(in));
     rep(i,in.size()) in[i].second *= -1;
 
     rep(i,in.size()){
         if(v.empty()) v.emplace_back(in[i]);
         else{
             pair<int, int> &u = v.back();
             int tmp = intervalState(in[i].first,in[i].second,u.first,u.second);
             switch (tmp){
                 case 0:
                 case 1:
                     v.emplace_back(in[i]);
                     break;
                 case 2:
                     u.first = in[i].first;
                     u.second = in[i].second;
                     break;
                 case 3:
                     break;
                 case 4:
                 case 5:
                     u.first = min(u.first, in[i].first);
                     u.second = max(u.second, in[i].second);
                     break;
                 case -1:
                     assert(0);
             }
         }
     }
     sort(all(v));
     return v;
 }

snippet     closedLoop
abbr        閉路の検出
 const int MAX_V = 505;
 
 vector<int> g[MAX_V];
 vector<int> tp;
 
 bool visit(int v, vector<int> &color){
     color[v] = 1;
     rep(i,g[v].size()){
         int d = g[v][i];
         if(color[d] == 2) continue;
         if(color[d] == 1) return false;
         if(not visit(d, color)) return false;
     }
     tp.emplace_back(v);
     color[v] = 2;
     return true;
 }
 
 bool topologicalSort(int v){
     vector<int> color(v);
     rep(i,v){
         if(not color[i] && not visit(i, color)) return false;
     }
     reverse(all(tp));
     return true;
 }

snippet     treeDP
abbr        全方位木DPによる木の直径の演算
 struct edge {
     int to, cost;
 };
 
 vector< edge > g[100000];
 long long dist[100000];
 
 
 void dfs1(int idx, int par) {
     for(edge &e : g[idx]) {
         if(e.to == par) continue;
         dfs1(e.to, idx);
         dist[idx] = max(dist[idx], dist[e.to] + e.cost);
     }
 }
 
 int dfs2(int idx, int d_par, int par) {
     vector< pair< int, int > > d_child;
     d_child.emplace_back(0, -1); // 番兵みたいなアレ
     for(edge &e : g[idx]) {
         if(e.to == par) d_child.emplace_back(d_par + e.cost, e.to);
         else d_child.emplace_back(e.cost + dist[e.to], e.to);
     }
     sort(d_child.rbegin(), d_child.rend());
     int ret = d_child[0].first + d_child[1].first; // 最大から 2 つ
     for(edge &e : g[idx]) {
         if(e.to == par) continue;
         // 基本は d_child() の最大が d_par になるが, e.to の部分木が最大値のときはそれを取り除く必要がある
         ret = max(ret, dfs2(e.to, d_child[d_child[0].second == e.to].first, idx));
     }
     return (ret);
 }
 
 int solve(int v/*頂点数*/){
     dfs1(v / 2, - 1);
     return dfs(v / 2, 0, - 1);
 }
