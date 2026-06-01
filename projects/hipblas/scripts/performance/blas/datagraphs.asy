import graph;
import utils;
import stats;

//asy datagraphs -u "xlabel=\"\$\\bm{u}\cdot\\bm{B}/uB$\"" -u "doyticks=false" -u "ylabel=\"\"" -u "legendlist=\"a,b\""

texpreamble("\usepackage{bm}");

size(400, 300, IgnoreAspect);

scale(Linear,Linear);
//scale(Log, Log);
//scale(Log,Linear);

bool dolegend = true;

string filenames = "";
string legendlist = "";

real xmin = -inf;
real xmax = inf;

bool doxticks = true;
bool doyticks = true;
string xlabel = "Problem size N";
string ylabel = "Time [s]";

bool normalize = false;

bool raw = true;
real theoMax = -1;
int speedup = 2;
int colnum=1;
usersetting();
write("filenames:\"", filenames+"\"");

if(filenames == "")
    filenames = getstring("filenames");

if (legendlist == "")
    legendlist=filenames;
bool myleg = ((legendlist == "") ? false: true);
string[] legends=set_legends(legendlist);

if(normalize) {
   scale(Log, Linear);
   ylabel = "Time / problem size N, [s]";
}

bool plotxval(real x) {
   return x >= xmin && x <= xmax;
}

string[] listfromcsv(string input)
{
    string list[] = new string[];
    int n = -1;
    bool flag = true;
    int lastpos;
    while(flag) {
        ++n;
        int pos = find(input, ",", lastpos);
        string found;
        if(lastpos == -1) {
            flag = false;
            found = "";
        }
        found = substr(input, lastpos, pos - lastpos);
        if(flag) {
            list.push(found);
            lastpos = pos > 0 ? pos + 1 : -1;
        }
    }
    return list;
}

string[] testlist = listfromcsv(filenames);

real[][] x = new real[testlist.length][];
real[][] y = new real[testlist.length][];
real[][] ly = new real[testlist.length][];
real[][] hy = new real[testlist.length][];
real[][][] data = new real[testlist.length][][];
real xmax = 0.0;
real xmin = inf;

for(int n = 0; n < testlist.length; ++n)
{
    string filename = testlist[n];

    data[n] = new real[][];
    write(filename);

    real[] ly;
    real[] hy;

    int dataidx = 0;

    bool moretoread = true;
    file fin = input(filename);
    while(moretoread) {
        int a = fin;
        if(a == 0) {
            moretoread = false;
            break;
        }

        int N = fin;
        //Flush out time

        for(int i = 0; i < N; ++i) {
            real temp = fin;
        }

        N = fin;

        if (N > 0) {
            xmax = max(a,xmax);
            xmin = min(a,xmin);

            x[n].push(a);

            data[n][dataidx] = new real[N];

            real vals[] = new real[N];
            for(int i = 0; i < N; ++i) {
                vals[i] = fin;
                data[n][dataidx][i] = vals[i];
            }

            real[] medlh = mediandev(vals);
            y[n].push(medlh[0]);
            ly.push(medlh[1]);
            hy.push(medlh[2]);
            ++dataidx;
        }

        //Flush out bandwidth
        N = fin;
        for(int i = 0; i < N; ++i) {
            real temp = fin;
        }
    }

    pen graphpen = Pen(n);
    if(n == 2)
        graphpen = darkgreen;

    pair[] z;
    pair[] dp;
    pair[] dm;
    for(int i = 0; i < x[n].length; ++i) {
        if(plotxval(x[n][i])) {
            z.push((x[n][i] , y[n][i]));
            dp.push((0 , y[n][i] - hy[i]));
            dm.push((0 , y[n][i] - ly[i]));
        }
    }
    errorbars(z, dp, dm, graphpen);

    guide g = scale(0.5mm) * unitcircle;
    marker mark = marker(g, Draw(graphpen + solid));

    bool drawme[] = new bool[x[n].length];
    for(int i = 0; i < drawme.length; ++i) {
        drawme[i] = true;
        if(!plotxval(x[n][i]))
	    drawme[i] = false;
        if(y[n][i] <= 0.0)
	    drawme[i] = false;
    }

    draw(graph(x[n], y[n], drawme), graphpen,
         myleg ? legends[n] : texify(filename), mark);
}

if( theoMax!=-1 )
{
    pen graphpen = Pen(testlist.length);
    guide g = scale(0.5mm) * unitcircle;
    marker mark = marker(g, Draw(graphpen + solid));

    draw((0,theoMax)--(x[0][x[0].length-1],theoMax),graphpen+dashed,"Theoretical Peak Performance: "+ format( "%.0f", theoMax) + " GFLOPS/s", mark);
    yaxis(ylabel,ymax=theoMax+theoMax/10,axis=LeftRight,ticks=RightTicks);
}


if(doxticks)
   xaxis(xlabel,BottomTop,LeftTicks);
else
    xaxis(xlabel);

if(doyticks)
    yaxis(ylabel,speedup > 1 ? Left : LeftRight,RightTicks);
else
   yaxis(ylabel,LeftRight);


if(dolegend)
    attach(legend(),point(plain.E),(speedup > 1 ? 60*plain.E + 40 *plain.N : 20*plain.E)  );


if(speedup > 1) {
    string[] legends = listfromcsv(legendlist);
    // TODO: error bars
    // TODO: when there is data missing at one end, the axes might be weird

    picture secondary=secondaryY(new void(picture pic) {
        scale(pic,Linear,Linear);
        real ymin = inf;
        real ymax = -inf;
	    int penidx = testlist.length + 1; //1 more than theoMax
        for(int n = 0; n < testlist.length; n += speedup) {

	        for(int next = 1; next < speedup; ++next) {
                real[] baseval = new real[];
                real[] yval = new real[];
                pair[] zy;
                pair[] dp;
                pair[] dm;

                for(int i = 0; i < x[n].length; ++i) {
                    for(int j = 0; j < x[n+next].length; ++j) {
                        if (x[n][i] == x[n+next][j]) {
                            baseval.push(x[n][i]);
                            real val = y[n][i] / y[n+next][j];
                            yval.push(val);

                            zy.push((x[n][i], val));
                            real[] lowhi = ratiodev(data[n][i], data[n+next][j]);
                            real hi = lowhi[1];
                            real low = lowhi[0];

                            dp.push((0 , hi - val));
                            dm.push((0 , low - val));

                            ymin = min(val, ymin);
                            ymax = max(val, ymax);
                            break;
                        }
                    }
                }

                if(baseval.length > 0){
                    pen p = Pen(penidx)+dashed;
                    ++penidx;

                    guide g = scale(0.5mm) * unitcircle;
                    marker mark = marker(g, Draw(p + solid));

                    draw(pic,graph(pic,baseval, yval),p,legends[n] + " vs " + legends[n+next],mark);
                    errorbars(pic, zy, dp, dm, p);
                }

                {
                    real[] fakex = {xmin, xmax};
                    real[] fakey = {ymin, ymax};
                    // draw an invisible graph to set up the limits correctly.
                    draw(pic,graph(pic,fakex, fakey),invisible);
                }

	        }
        }

	    yequals(pic, 1.0, lightgrey);
        yaxis(pic,"speedup",Right,  black,LeftTicks);
        attach(legend(pic),point(plain.E), 60*plain.E - 40 *plain.N  );
    });


    add(secondary);
}
