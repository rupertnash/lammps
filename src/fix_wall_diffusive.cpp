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
#include "random_mars.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{XLO=0,XHI=1,YLO=2,YHI=3,ZLO=4,ZHI=5};
enum{NONE=0,EDGE,CONSTANT,VARIABLE};

/* ---------------------------------------------------------------------- */

FixWallDiffusive::FixWallDiffusive(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  nwall(0),
  wall_temp(-1), TMAC(0.0), prng(NULL)
{
  if (narg < 4) error->all(FLERR,"Illegal fix wall/diffusive command");

  dynamic_group_allow = 1;

  // parse args

  nwall = 0;
  int scaleflag = 1;
  // PRNG seed, must be > 0
  int seed = 0;
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
      wall_temp = force->numeric(FLERR, arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "TMAC") == 0) {
      TMAC = force->numeric(FLERR, arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "seed") == 0) {
      seed = force->inumeric(FLERR, arg[iarg+1]);
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

  if (wall_temp <= 0.0) {
    error->all(FLERR,
	       "Must specify temperature > 0 K for fix wall/diffusive");
  }

  if (TMAC <= 0.0 || TMAC > 1.0) {
    error->all(FLERR,
	       "Tangential momentum accommodation coefficient must be in [0.0, 1.0] for fix wall/diffusive");
  }

  if (seed <= 0) {
    error->all(FLERR,
	       "Pseudorandom number generator seed must be positive integer");
  }

  prng = new RanMars(lmp, seed + comm->me);

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
      else coord0[m] *= zscale;
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
		   "diffusive walls");
}

namespace LAMMPS_NS {
  struct DiffusiveWallImpl {
    // Entry point 
    void foreach_atom(int boxdim, int wallwhich_m, double coord) {
      switch (boxdim) {
      case 2:
	foreach_atom_d<2>(wallwhich_m, coord);
	break;
      case 3:
	foreach_atom_d<3>(wallwhich_m, coord);
	break;
      default:
	error->all(FLERR, "Invalid dimensionality for fix wall/diffusive");
      }
    }

    template<int boxdim>
    void foreach_atom_d(int wallwhich_m, double coord) {
      // Dispatch on the dim to the bounded set of boundary directions
      // to the templated implementations to avoid any overhead
      const int dim = wallwhich_m / 2;
      const bool side = wallwhich_m % 2;
      switch (dim) {
      case 0:
	foreach_atom_dw<boxdim, 0>(side, coord);
	break;
      case 1:
	foreach_atom_dw<boxdim, 1>(side, coord);
	break;
      case 2:
	foreach_atom_dw<boxdim, 2>(side, coord);
	break;
      default:
	error->all(FLERR, "Invalid wall in fix wall/diffusive");
      }
    }
    template<int boxdim, int dim>
    void foreach_atom_dw(bool side, double coord) {
      if (side)
	foreach_impl<boxdim, dim, true>(coord);
      else
	foreach_impl<boxdim, dim, false>(coord);
    }

    // Loop over all local atoms and, if the mask says so, do the
    // update
    template <int boxdim, int dim, bool is_upper>
    void foreach_impl(double coord) {
      for (int i = 0; i < atom->nlocal; i++)
	if (atom->mask[i] & diff->groupbit) {
	  update<boxdim, dim, is_upper>(i, coord);
	}
    }
    
    template <int boxdim, int dim, bool is_upper>
    void update(const int i, double coord) {
      double** x = atom->x;
      if ((is_upper && (x[i][dim] > coord))
	  || (!is_upper && (x[i][dim] < coord) )) {
	// Update position as for reflection
	x[i][dim] = coord - (x[i][dim] - coord);
	
	// update velocity
	if (diff->prng->uniform() <= diff->TMAC) {
	  // Diffusive with prob(TMAC)
	  diffusive_update<boxdim, dim, is_upper>(i);
	} else {
	  // Reflective with prob(1-TMAC)
	  reflective_update<boxdim, dim, is_upper>(i);
	}
      }
    }

    template <int boxdim, int dim, bool is_upper>
    void reflective_update(const int i) {
      atom->v[i][dim] *= -1.0;
    }
        
    template <int boxdim, int dim, bool is_upper>
    void diffusive_update(const int i) {
      const double wall_temp = diff->wall_temp;
      // Standard deviation of the Gaussian distribution that we draw
      // sigma^2 = kT/m
      const double sigma2 = force->boltz * diff->wall_temp / (atom->mass[atom->type[i]] * force->mvv2e);
      const double sigma = sqrt(sigma2);

      for (int d = 0; d < boxdim; ++d) {
	atom->v[i][d] = (d == dim) ?
	  draw_perpendicular(sigma2) :
	  draw_tangential(sigma);
      }
    }

    double draw_perpendicular(double sigma2) {
      // Inverse transform sampling - see Wikipedia!
      // Need CDF (cumulative distrubution function)

      // Our PDF is: P(v) = v *exp(-v^2/(2 sigma^2))/sigma^2
      // Checked form with wolfram alpha:
      // integrate v *exp(-v^2/(2 sigma^2))/sigma^2 dv from v=0 to infinity
      // gives 1 as required

      // CDF(v) = integrate P(v') dv' from v'=0 to v
      //        = 1 - exp(-v^2 / (2 sigma^2))

      // Now draw uniform deviates (u) in [0,1] and invert CDF(v) = u
      // v = sqrt(-2 sigma^2 ln(1-u))
      double u = diff->prng->uniform();
      return sqrt(-2.0 * sigma2 * log(1.0 - u));
    }
    double draw_tangential(double sigma) {
      // Gaussian distribution:
      // P(x) = exp(-x^2 / (2 sigma^2) ) / sqrt(2 pi sigma^2)
      return sigma * diff->prng->gaussian();
    }

    DiffusiveWallImpl(FixWallDiffusive* diff_, Atom* atom_, Error* error_, Force* force_)
      :
      diff(diff_), atom(atom_), error(error_), force(force_)
    {
    }
    
    FixWallDiffusive* diff;
    Atom* atom;
    Error* error;
    Force* force;
  };
}


void FixWallDiffusive::post_integrate()
{
  if (varflag) modify->clearstep_compute();

  DiffusiveWallImpl updater(this, atom, error, force);
  for (int m = 0; m < nwall; m++) {
    // coord = current position of wall
    // evaluate variable if necessary, wrap with clear/add
    double coord;
    if (wallstyle[m] == VARIABLE) {
      coord = input->variable->compute_equal(varindex[m]);
      if (wallwhich[m] < YLO) coord *= xscale;
      else if (wallwhich[m] < ZLO) coord *= yscale;
      else coord *= zscale;
    } else coord = coord0[m];

    updater.foreach_atom(domain->dimension, wallwhich[m], coord);
  }
  if (varflag) modify->addstep_compute(update->ntimestep + 1);
}

