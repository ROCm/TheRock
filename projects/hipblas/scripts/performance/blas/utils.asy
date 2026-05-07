// Find the comma-separated strings to use in the legend
string[] set_legends(string runlegs)
{
   string[] legends;
   bool myleg=((runlegs== "") ? false: true);
   bool flag=true;
   int n=-1;
   int lastpos=0;
   string legends[];
   if(myleg) {
      string runleg;
      while(flag) {
	 ++n;
	 int pos=find(runlegs,",",lastpos);
	 if(lastpos == -1) {runleg=""; flag=false;}

	 runleg=substr(runlegs,lastpos,pos-lastpos);

	 lastpos=pos > 0 ? pos+1 : -1;
	 if(flag) legends.push(runleg);
      }
   }
   return legends;
}
