/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
   ------------------------------------------------------------------------- */

#include <stdlib.h>
#include <string.h>
#include "fix_wall_diffusive.h"
#include "atom.h"
#include "comm.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "lattice.h"
#include "input.h"
#include "variable.h"
#include "error.h"
#include "force.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{XLO=0,XHI=1,YLO=2,YHI=3,ZLO=4,ZHI=5};
enum{NONE=0,EDGE,CONSTANT,VARIABLE};

/* ---------------------------------------------------------------------- */

namespace {
  double const dbcR=8.315; // J/(mol*K)
  double const dbcpi=3.1415926;
  double const dbckb=1.3807E-23; // J/K
  double const dbcM=39.96; // g/mol  ////////////////////////////////May be changed-1/6////////////////////////////
  double const dbcNa=6.02E23;
  double const dbcm=dbcM/dbcNa/1000; //kg
}

FixWallDiffusive::FixWallDiffusive(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  nwall(0),
  tempK(-1), TMAC(1.0)
{
  if (narg < 4) error->all(FLERR,"Illegal fix wall/diffusive command");

  dynamic_group_allow = 1;

  // parse args

  nwall = 0;
  int scaleflag = 1;

  int iarg = 3;
  while (iarg < narg) {
    if ((strcmp(arg[iarg],"xlo") == 0) || (strcmp(arg[iarg],"xhi") == 0) ||
        (strcmp(arg[iarg],"ylo") == 0) || (strcmp(arg[iarg],"yhi") == 0) ||
        (strcmp(arg[iarg],"zlo") == 0) || (strcmp(arg[iarg],"zhi") == 0)) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix wall/diffusive command");

      int newwall;
      if (strcmp(arg[iarg],"xlo") == 0) newwall = XLO;
      else if (strcmp(arg[iarg],"xhi") == 0) newwall = XHI;
      else if (strcmp(arg[iarg],"ylo") == 0) newwall = YLO;
      else if (strcmp(arg[iarg],"yhi") == 0) newwall = YHI;
      else if (strcmp(arg[iarg],"zlo") == 0) newwall = ZLO;
      else if (strcmp(arg[iarg],"zhi") == 0) newwall = ZHI;

      for (int m = 0; (m < nwall) && (m < 6); m++)
        if (newwall == wallwhich[m])
          error->all(FLERR,"Wall defined twice in fix wall/diffusive command");

      wallwhich[nwall] = newwall;
      if (strcmp(arg[iarg+1],"EDGE") == 0) {
        wallstyle[nwall] = EDGE;
        int dim = wallwhich[nwall] / 2;
        int side = wallwhich[nwall] % 2;
        if (side == 0) coord0[nwall] = domain->boxlo[dim];
        else coord0[nwall] = domain->boxhi[dim];
      } else if (strstr(arg[iarg+1],"v_") == arg[iarg+1]) {
        wallstyle[nwall] = VARIABLE;
        int n = strlen(&arg[iarg+1][2]) + 1;
        varstr[nwall] = new char[n];
        strcpy(varstr[nwall],&arg[iarg+1][2]);
      } else {
        wallstyle[nwall] = CONSTANT;
        coord0[nwall] = force->numeric(FLERR,arg[iarg+1]);
      }

      nwall++;
      iarg += 2;

    } else if (strcmp(arg[iarg],"units") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal wall/diffusive command");
      if (strcmp(arg[iarg+1],"box") == 0) scaleflag = 0;
      else if (strcmp(arg[iarg+1],"lattice") == 0) scaleflag = 1;
      else error->all(FLERR,"Illegal fix wall/diffusive command");
      iarg += 2;
    } else if (strcmp(arg[iarg], "walltemp") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix wall/diffusive command");
      tempK = strtod(arg[iarg+1], NULL);
      iarg += 2;
    } else if (strcmp(arg[iarg], "TMAC") == 0) {
      tmac = strtod(arg[iarg+1], NULL);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix wall/diffusive command");
  }

  // error check

  if (nwall == 0) error->all(FLERR,"Illegal fix wall command");

  for (int m = 0; m < nwall; m++) {
    if ((wallwhich[m] == XLO || wallwhich[m] == XHI) && domain->xperiodic)
      error->all(FLERR,"Cannot use fix wall/diffusive in periodic dimension");
    if ((wallwhich[m] == YLO || wallwhich[m] == YHI) && domain->yperiodic)
      error->all(FLERR,"Cannot use fix wall/diffusive in periodic dimension");
    if ((wallwhich[m] == ZLO || wallwhich[m] == ZHI) && domain->zperiodic)
      error->all(FLERR,"Cannot use fix wall/diffusive in periodic dimension");
  }

  for (int m = 0; m < nwall; m++)
    if ((wallwhich[m] == ZLO || wallwhich[m] == ZHI) && domain->dimension == 2)
      error->all(FLERR,
                 "Cannot use fix wall/diffusive zlo/zhi for a 2d simulation");

  if (tempK <= 0.0) {
    error->all(FLERR,
	       "Must specify temperature > 0 K for fix wall/diffusive");
  }

  if (TMAC < 0.0 || TMAC > 1.0) {
    error->all(FLERR,
	       "Tangential momentum accommodation coefficient must be in [0.0, 1.0] for fix wall/diffusive");
    // scale factors for CONSTANT and VARIABLE walls

    int flag = 0;
    for (int m = 0; m < nwall; m++)
      if (wallstyle[m] != EDGE) flag = 1;

    if (flag) {
      if (scaleflag) {
	xscale = domain->lattice->xlattice;
	yscale = domain->lattice->ylattice;
	zscale = domain->lattice->zlattice;
      }
      else xscale = yscale = zscale = 1.0;

      for (int m = 0; m < nwall; m++) {
	if (wallstyle[m] != CONSTANT) continue;
	if (wallwhich[m] < YLO) coord0[m] *= xscale;
	else if (wallwhich[m] < ZLO) coord0[m] *= yscale;
	f      else coord0[m] *= zscale;
      }
    }

    // set varflag if any wall positions are variable

    varflag = 0;
    for (int m = 0; m < nwall; m++)
      if (wallstyle[m] == VARIABLE) varflag = 1;
  }

  /* ---------------------------------------------------------------------- */

  FixWallDiffusive::~FixWallDiffusive()
  {
    if (copymode) return;

    for (int m = 0; m < nwall; m++)
      if (wallstyle[m] == VARIABLE) delete [] varstr[m];
  }

  /* ---------------------------------------------------------------------- */

  int FixWallDiffusive::setmask()
  {
    int mask = 0;
    mask |= POST_INTEGRATE;
    mask |= POST_INTEGRATE_RESPA;
    return mask;
  }

  /* ---------------------------------------------------------------------- */

  void FixWallDiffusive::init()
  {
    for (int m = 0; m < nwall; m++) {
      if (wallstyle[m] != VARIABLE) continue;
      varindex[m] = input->variable->find(varstr[m]);
      if (varindex[m] < 0)
	error->all(FLERR,"Variable name for fix wall/diffusive does not exist");
      if (!input->variable->equalstyle(varindex[m]))
	error->all(FLERR,"Variable for fix wall/diffusive is invalid style");
    }

    int nrigid = 0;
    for (int i = 0; i < modify->nfix; i++)
      if (modify->fix[i]->rigid_flag) nrigid++;

    if (nrigid && comm->me == 0)
      error->warning(FLERR,"Should not allow rigid bodies to bounce off "
		     "relecting walls");
  }

  /* ---------------------------------------------------------------------- */

  void FixWallDiffusive::post_integrate()
  {
    int i,dim,side;
    double coord;

    // coord = current position of wall
    // evaluate variable if necessary, wrap with clear/add

    double **x = atom->x;
    double **v = atom->v;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
   
    double diff1=1.0;


    if (varflag) modify->clearstep_compute();

    for (int m = 0; m < nwall; m++) {
      if (wallstyle[m] == VARIABLE) {
	coord = input->variable->compute_equal(varindex[m]);
	if (wallwhich[m] < YLO) coord *= xscale;
	else if (wallwhich[m] < ZLO) coord *= yscale;
	else coord *= zscale;
      } else coord = coord0[m];

      dim = wallwhich[m] / 2;
      side = wallwhich[m] % 2;

      for (i = 0; i < nlocal; i++)
	if (mask[i] & groupbit) {
	  if (side == 0) {
	    if (x[i][dim] < coord) {
	      x[i][dim] = coord + (coord - x[i][dim]);

	      // **************************the wall at low edge in z direction*******************************
	      double W  = (double)rand() / (RAND_MAX + 1.0);
	      // *********************************************************
	      if(W<=TMAC){	
		double dbcvp;// most probable velocity
		dbcvp=sqrt(2*dbcR*tempK/dbcM*1000);
		double dbcvm; // m/s
		//take care for *dbcvp	
		double dbcvz;
		double dbcvcut=3.0;/////////////////////////////////May be changed-3/6////////////////////////////
		double dbcpa;	
		double dbcpth;  // probablity threshold
		double dbcfvmax;
	
		int setbz=0;
		dbcvm=dbcvp*sqrt(2)/2; // the velcoty which gives the maximum value of the fv 
		dbcfvmax =dbcm/dbckb/tempK*dbcvm*exp(-dbcm*dbcvm*dbcvm/2/dbckb/tempK)*dbcvp; 
		while(setbz==0){

		  dbcvz=(double)rand() / RAND_MAX *dbcvcut*dbcvp;
		  dbcpa=(double)rand() / RAND_MAX*dbcfvmax;
		  dbcpth=dbcm/dbckb/tempK*dbcvz*exp(-dbcm*dbcvz*dbcvz/2/dbckb/tempK)*dbcvp;

		  if (dbcpa<=dbcpth){
		    v[i][dim]=dbcvz/100000;  // take care of /100000 from SI units to REAL units
		    setbz=1;		
		  }
		}
		/* -----------------------------------vx----------------------------------- */    
		
		dbcvm=1/dbcvp/sqrt(dbcpi); 
		// the velcoty which gives the maximum value of the fv ////////////////////////////////
		dbcfvmax =dbcvm*exp(-dbcm*dbcvm*dbcvm/2/dbckb/tempK);
		int setbx=0;
		while(setbx==0){		
		  dbcvz=dbcvcut*dbcvp*((double)rand()/RAND_MAX*2-1);
		  dbcpa=(double)rand()/RAND_MAX*dbcfvmax;
		  dbcpth=dbcvm*exp(-dbcm*dbcvz*dbcvz/2/dbckb/tempK);

		  if (dbcpa<=dbcpth){
		    v[i][0]=dbcvz/100000;  // take care of /100000 from SI units to REAL units
		    setbx=1;		
		  }
		}
		/* -----------------------------------vy----------------------------------- */		
		int setby=0;
		while(setby==0){   
		  dbcvz=dbcvcut*dbcvp*((double)rand()/RAND_MAX*2-1);
		  dbcpa=(double)rand()/RAND_MAX*dbcfvmax;
		  dbcpth=dbcvm*exp(-dbcm*dbcvz*dbcvz/2/dbckb/tempK);

		  if (dbcpa<=dbcpth){
		    v[i][1]=dbcvz/100000;  // take care of /100000 from SI units to REAL units
		    setby=1;		
		  }
		}

	      }else{ // W > TMAC //Give a specular reflection ////////////////#############////////////////////

		v[i][dim]=-v[i][dim];
	      }
	    }  //end   if (x[i][dim] < coord) 
	  } else {  // end  if (side == 0) 
	    if (x[i][dim] > coord) {
	      x[i][dim] = coord - (x[i][dim] - coord);
	      // **************************the wall at high edge in z direction*******************************
	      double W  = (double)rand() / (RAND_MAX + 1.0);
	      // ********************************************************* 
	      if(W<=TMAC){
		/* ---------------------------------vz------------------------------------- */
		double dbcvp;// most probable velocity
		dbcvp=sqrt(2*dbcR*tempK/dbcM*1000);
		double dbcvm; // m/s

		//take care for *dbcvp	
		double dbcvz;
		double dbcvcut=3.0; /////////////////////////////////May be changed-6/6////////////////////////////
		double dbcpa;
		double dbcpth;  // probablity threshold
		double dbcfvmax;

		dbcvm=dbcvp*sqrt(2)/2; // the velcoty which gives the maximum value of the fv 
		dbcfvmax =dbcm/dbckb/tempK*dbcvm*exp(-dbcm*dbcvm*dbcvm/2/dbckb/tempK)*dbcvp; 
		int setuz=0;
		while(setuz==0){
		  dbcvz=(double)rand() / RAND_MAX *dbcvcut*dbcvp;
		  dbcpa=(double)rand() / RAND_MAX*dbcfvmax;
		  dbcpth=dbcm/dbckb/tempK*dbcvz*exp(-dbcm*dbcvz*dbcvz/2/dbckb/tempK)*dbcvp;
		  if (dbcpa<=dbcpth){
		    v[i][dim]=-dbcvz/100000; // take care of /100000 from SI units to REAL units
		    setuz=1;
		  }
		}
		/* -----------------------------------vx----------------------------------- */    
		dbcvm=1/dbcvp/sqrt(dbcpi); 
		// the velcoty which gives the maximum value of the fv ////////////////////////////////
		dbcfvmax =dbcvm*exp(-dbcm*dbcvm*dbcvm/2/dbckb/tempK);		

		int setbx=0;
		while(setbx==0){ 		
		  dbcvz=dbcvcut*dbcvp*((double)rand()/RAND_MAX*2-1);
		  dbcpa=(double)rand()/RAND_MAX*dbcfvmax;
		  dbcpth=dbcvm*exp(-dbcm*dbcvz*dbcvz/2/dbckb/tempK);

		  if (dbcpa<=dbcpth){
		    v[i][0]=dbcvz/100000;  // take care of /100000 from SI units to REAL units
		    setbx=1;		
		  }
		}
		/* -----------------------------------vy----------------------------------- */
		int setby=0;
		while(setby==0){  
		  dbcvz=dbcvcut*dbcvp*((double)rand()/RAND_MAX*2-1);
		  dbcpa=(double)rand()/RAND_MAX*dbcfvmax;
		  dbcpth=dbcvm*exp(-dbcm*dbcvz*dbcvz/2/dbckb/tempK);

		  if (dbcpa<=dbcpth){
		    v[i][1]=dbcvz/100000;  // take care of /100000 from SI units to REAL units
		    setby=1;		
		  }
		}
	      }else{   // W > TMAC //Give a specular reflection ////////////////#############////////////////////
		v[i][dim]=-v[i][dim];
	      }
	      /* ---------------------------------------------------------------------- */ 
	    } // end if(x[i][dim]>coord)
	  } //} else {  // end  if (side == 0)  
	}// end       if (mask[i] & groupbit) 
    } //   end    if (mask[i] & groupbit) 
    if (varflag) modify->addstep_compute(update->ntimestep + 1);
  }

