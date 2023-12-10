#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    mover_PC_gpu(part, field, grd, param);
    return(0);
}

int mover_PC_cpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "*** CPU MOVER with SUBCYCLYING " << param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart)param->dt / ((double)part->n_sub_cycles);
    FPpart dto2 = .5 * dt_sub_cycling, qomdt2 = part->qom * dto2 / param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;

    // local (to the particle) electric and magnetic field
    FPfield Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

    // interpolation densities
    int ix, iy, iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];

    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    // start subcycling
    for (int i_sub = 0; i_sub < part->n_sub_cycles; i_sub++) {
        // move each particle with new fields
        for (int i = 0; i < part->nop; i++) {
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for (int innter = 0; innter < part->NiterMover; innter++) {
                // interpolation G-->P
                ix = 2 + int((part->x[i] - grd->xStart) * grd->invdx);
                iy = 2 + int((part->y[i] - grd->yStart) * grd->invdy);
                iz = 2 + int((part->z[i] - grd->zStart) * grd->invdz);

                // calculate weights
                xi[0] = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0] = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1] = grd->XN[ix][iy][iz] - part->x[i];
                eta[1] = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

                // set to zero local electric and magnetic field
                Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++) {
                            Exl += weight[ii][jj][kk] * field->Ex[ix - ii][iy - jj][iz - kk];
                            Eyl += weight[ii][jj][kk] * field->Ey[ix - ii][iy - jj][iz - kk];
                            Ezl += weight[ii][jj][kk] * field->Ez[ix - ii][iy - jj][iz - kk];
                            Bxl += weight[ii][jj][kk] * field->Bxn[ix - ii][iy - jj][iz - kk];
                            Byl += weight[ii][jj][kk] * field->Byn[ix - ii][iy - jj][iz - kk];
                            Bzl += weight[ii][jj][kk] * field->Bzn[ix - ii][iy - jj][iz - kk];
                        }

                // end interpolation
                omdtsq = qomdt2 * qomdt2 * (Bxl * Bxl + Byl * Byl + Bzl * Bzl);
                denom = 1.0 / (1.0 + omdtsq);
                // solve the position equation
                ut = part->u[i] + qomdt2 * Exl;
                vt = part->v[i] + qomdt2 * Eyl;
                wt = part->w[i] + qomdt2 * Ezl;
                udotb = ut * Bxl + vt * Byl + wt * Bzl;
                // solve the velocity equation
                uptilde = (ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) * denom;
                vptilde = (vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) * denom;
                wptilde = (wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) * denom;
                // update position
                part->x[i] = xptilde + uptilde * dto2;
                part->y[i] = yptilde + vptilde * dto2;
                part->z[i] = zptilde + wptilde * dto2;


            } // end of iteration
            // update the final position and velocity
            part->u[i] = 2.0 * uptilde - part->u[i];
            part->v[i] = 2.0 * vptilde - part->v[i];
            part->w[i] = 2.0 * wptilde - part->w[i];
            part->x[i] = xptilde + uptilde * dt_sub_cycling;
            part->y[i] = yptilde + vptilde * dt_sub_cycling;
            part->z[i] = zptilde + wptilde * dt_sub_cycling;


            //////////
            //////////
            ////////// BC

            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx) {
                if (param->PERIODICX == true) { // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                }
                else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2 * grd->Lx - part->x[i];
                }
            }

            if (part->x[i] < 0) {
                if (param->PERIODICX == true) { // PERIODIC
                    part->x[i] = part->x[i] + grd->Lx;
                }
                else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }


            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly) {
                if (param->PERIODICY == true) { // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                }
                else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2 * grd->Ly - part->y[i];
                }
            }

            if (part->y[i] < 0) {
                if (param->PERIODICY == true) { // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                }
                else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }

            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz) {
                if (param->PERIODICZ == true) { // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                }
                else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2 * grd->Lz - part->z[i];
                }
            }

            if (part->z[i] < 0) {
                if (param->PERIODICZ == true) { // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                }
                else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }



        }  // end of subcycling
    } // end of one particle

    return(0); // exit succcesfully
} // end of the mover

int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "*** GPU MOVER with SUBCYCLYING " << param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    struct particles gpu_part;
    struct EMfield   gpu_field;
    struct grid      gpu_grd;

    memcpy(&gpu_part,  part,  sizeof(particles));
    memcpy(&gpu_field, field, sizeof(EMfield));
    memcpy(&gpu_grd,   grd,   sizeof(grid));

    // GPU allocation for particles
    cudaMalloc(&gpu_part.x, sizeof(FPpart) * gpu_part.npmax);
    cudaMalloc(&gpu_part.y, sizeof(FPpart) * gpu_part.npmax);
    cudaMalloc(&gpu_part.z, sizeof(FPpart) * gpu_part.npmax);
    cudaMalloc(&gpu_part.u, sizeof(FPpart) * gpu_part.npmax);
    cudaMalloc(&gpu_part.v, sizeof(FPpart) * gpu_part.npmax);
    cudaMalloc(&gpu_part.w, sizeof(FPpart) * gpu_part.npmax);

    // GPU allocation for field
    cudaMalloc(&gpu_field.Bxn, sizeof(FPfield) * gpu_grd.nxn * gpu_grd.nyn * gpu_grd.nzn);
    cudaMalloc(&gpu_field.Byn, sizeof(FPfield) * gpu_grd.nxn * gpu_grd.nyn * gpu_grd.nzn);
    cudaMalloc(&gpu_field.Bzn, sizeof(FPfield) * gpu_grd.nxn * gpu_grd.nyn * gpu_grd.nzn);
    cudaMalloc(&gpu_field.Ex,  sizeof(FPfield) * gpu_grd.nxn * gpu_grd.nyn * gpu_grd.nzn);
    cudaMalloc(&gpu_field.Ey,  sizeof(FPfield) * gpu_grd.nxn * gpu_grd.nyn * gpu_grd.nzn);
    cudaMalloc(&gpu_field.Ez,  sizeof(FPfield) * gpu_grd.nxn * gpu_grd.nyn * gpu_grd.nzn);

    // GPU allocation for grid
    cudaMalloc(&gpu_grd.XN,  sizeof(FPfield) * gpu_grd.nxn * gpu_grd.nyn * gpu_grd.nzn);
    cudaMalloc(&gpu_grd.YN,  sizeof(FPfield) * gpu_grd.nxn * gpu_grd.nyn * gpu_grd.nzn);
    cudaMalloc(&gpu_grd.ZN,  sizeof(FPfield) * gpu_grd.nxn * gpu_grd.nyn * gpu_grd.nzn);

    // CPU-GPU transfer for particles
    cudaMemcpy(gpu_part.x, part->x, sizeof(FPpart) * gpu_part.npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_part.y, part->y, sizeof(FPpart) * gpu_part.npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_part.z, part->z, sizeof(FPpart) * gpu_part.npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_part.u, part->u, sizeof(FPpart) * gpu_part.npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_part.v, part->v, sizeof(FPpart) * gpu_part.npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_part.w, part->w, sizeof(FPpart) * gpu_part.npmax, cudaMemcpyHostToDevice);

    // CPU-GPU transfer for field
    cudaMemcpy(gpu_field.Bxn, field->Bxn_flat, sizeof(FPfield) * gpu_grd.nxn * gpu_grd.nyn * gpu_grd.nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_field.Byn, field->Byn_flat, sizeof(FPfield) * gpu_grd.nxn * gpu_grd.nyn * gpu_grd.nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_field.Bzn, field->Bzn_flat, sizeof(FPfield) * gpu_grd.nxn * gpu_grd.nyn * gpu_grd.nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_field.Ex,  field->Ex_flat,  sizeof(FPfield) * gpu_grd.nxn * gpu_grd.nyn * gpu_grd.nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_field.Ey,  field->Ey_flat,  sizeof(FPfield) * gpu_grd.nxn * gpu_grd.nyn * gpu_grd.nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_field.Ez,  field->Ez_flat,  sizeof(FPfield) * gpu_grd.nxn * gpu_grd.nyn * gpu_grd.nzn, cudaMemcpyHostToDevice);

    // CPU-GPU transfer for grid
    cudaMemcpy(gpu_grd.XN, grd->XN_flat, sizeof(FPfield) * gpu_grd.nxn * gpu_grd.nyn * gpu_grd.nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_grd.YN, grd->YN_flat, sizeof(FPfield) * gpu_grd.nxn * gpu_grd.nyn * gpu_grd.nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_grd.ZN, grd->ZN_flat, sizeof(FPfield) * gpu_grd.nxn * gpu_grd.nyn * gpu_grd.nzn, cudaMemcpyHostToDevice);

    // GPU execution
    int threads = 64;
    int blocks  = (part->nop + threads - 1) / threads;
    mover_PC_gpu_kernel<<<threads, blocks>>>(&gpu_part, &gpu_field, &gpu_grd, param);
    cudaDeviceSynchronize();

    // GPU-CPU transfer for particles
    cudaMemcpy(part->x, gpu_part.x, sizeof(FPpart) * part->npmax, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->y, gpu_part.y, sizeof(FPpart) * part->npmax, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->z, gpu_part.z, sizeof(FPpart) * part->npmax, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->u, gpu_part.u, sizeof(FPpart) * part->npmax, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->v, gpu_part.v, sizeof(FPpart) * part->npmax, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->w, gpu_part.w, sizeof(FPpart) * part->npmax, cudaMemcpyDeviceToHost);

    // GPU deallocation for particles
    cudaFree(gpu_part.x);
    cudaFree(gpu_part.y);
    cudaFree(gpu_part.z);
    cudaFree(gpu_part.u);
    cudaFree(gpu_part.v);
    cudaFree(gpu_part.w);

    // GPU deallocation for field
    cudaFree(gpu_field.Bxn);
    cudaFree(gpu_field.Byn);
    cudaFree(gpu_field.Bzn);
    cudaFree(gpu_field.Ex);
    cudaFree(gpu_field.Ey);
    cudaFree(gpu_field.Ez);

    // GPU deallocation for grid
    cudaFree(gpu_grd.XN);
    cudaFree(gpu_grd.YN);
    cudaFree(gpu_grd.ZN);

    return(0);
}

__global__ void mover_PC_gpu_kernel(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // each particle is moved in parallel by a separate thread
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // move each particle with new fields
    if (i < part->nop)
    {
        // auxiliary variables
        FPpart dt_sub_cycling = (FPpart)param->dt / ((double)part->n_sub_cycles);
        FPpart dto2 = .5 * dt_sub_cycling, qomdt2 = part->qom * dto2 / param->c;
        FPpart omdtsq, denom, ut, vt, wt, udotb;

        // local (to the particle) electric and magnetic field
        FPfield Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

        // interpolation densities
        int ix, iy, iz;
        FPfield weight[8];
        FPfield xi[2], eta[2], zeta[2];

        // intermediate particle position and velocity
        FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

        // start subcycling
        for (int i_sub = 0; i_sub < part->n_sub_cycles; i_sub++)
        {
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];

            // calculate the average velocity iteratively
            for (int innter = 0; innter < part->NiterMover; innter++)
            {
                // interpolation G-->P
                ix = 2 + int((part->x[i] - grd->xStart) * grd->invdx);
                iy = 2 + int((part->y[i] - grd->yStart) * grd->invdy);
                iz = 2 + int((part->z[i] - grd->zStart) * grd->invdz);

                // calculate weights
                xi[0]   = part->x[i] - grd->XN_flat[get_idx(ix - 1, iy, iz, grd->nyn, grd->nzn)];
                eta[0]  = part->y[i] - grd->YN_flat[get_idx(ix, iy - 1, iz, grd->nyn, grd->nzn)];
                zeta[0] = part->z[i] - grd->ZN_flat[get_idx(ix, iy, iz - 1, grd->nyn, grd->nzn)];
                xi[1]   = grd->XN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->x[i];
                eta[1]  = grd->YN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->y[i];
                zeta[1] = grd->ZN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[get_idx(ii, jj, kk, 2, 2)] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

                // set to zero local electric and magnetic field
                Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                        {
                            Exl += weight[get_idx(ii, jj, kk, 2, 2)] * field->Ex_flat[get_idx(ix - ii, iy - jj, iz - kk, grd->nyn, grd->nzn)];
                            Eyl += weight[get_idx(ii, jj, kk, 2, 2)] * field->Ey_flat[get_idx(ix - ii, iy - jj, iz - kk, grd->nyn, grd->nzn)];
                            Ezl += weight[get_idx(ii, jj, kk, 2, 2)] * field->Ez_flat[get_idx(ix - ii, iy - jj, iz - kk, grd->nyn, grd->nzn)];
                            Bxl += weight[get_idx(ii, jj, kk, 2, 2)] * field->Bxn_flat[get_idx(ix - ii, iy - jj, iz - kk, grd->nyn, grd->nzn)];
                            Byl += weight[get_idx(ii, jj, kk, 2, 2)] * field->Byn_flat[get_idx(ix - ii, iy - jj, iz - kk, grd->nyn, grd->nzn)];
                            Bzl += weight[get_idx(ii, jj, kk, 2, 2)] * field->Bzn_flat[get_idx(ix - ii, iy - jj, iz - kk, grd->nyn, grd->nzn)];
                        }

                // end interpolation
                omdtsq = qomdt2 * qomdt2 * (Bxl * Bxl + Byl * Byl + Bzl * Bzl);
                denom = 1.0 / (1.0 + omdtsq);
                // solve the position equation
                ut = part->u[i] + qomdt2 * Exl;
                vt = part->v[i] + qomdt2 * Eyl;
                wt = part->w[i] + qomdt2 * Ezl;
                udotb = ut * Bxl + vt * Byl + wt * Bzl;
                // solve the velocity equation
                uptilde = (ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) * denom;
                vptilde = (vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) * denom;
                wptilde = (wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) * denom;
                // update position
                part->x[i] = xptilde + uptilde * dto2;
                part->y[i] = yptilde + vptilde * dto2;
                part->z[i] = zptilde + wptilde * dto2;
            } // end of iteration

            // update the final position and velocity
            part->u[i] = 2.0 * uptilde - part->u[i];
            part->v[i] = 2.0 * vptilde - part->v[i];
            part->w[i] = 2.0 * wptilde - part->w[i];
            part->x[i] = xptilde + uptilde * dt_sub_cycling;
            part->y[i] = yptilde + vptilde * dt_sub_cycling;
            part->z[i] = zptilde + wptilde * dt_sub_cycling;

            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx)
            {
                if (param->PERIODICX == true) // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                else // REFLECTING BC
                {
                    part->u[i] = -part->u[i];
                    part->x[i] = 2 * grd->Lx - part->x[i];
                }
            }

            if (part->x[i] < 0)
            {
                if (param->PERIODICX == true) // PERIODIC
                    part->x[i] = part->x[i] + grd->Lx;
                else // REFLECTING BC
                {
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }

            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly)
            {
                if (param->PERIODICY == true) // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                else // REFLECTING BC
                {
                    part->v[i] = -part->v[i];
                    part->y[i] = 2 * grd->Ly - part->y[i];
                }
            }

            if (part->y[i] < 0)
            {
                if (param->PERIODICY == true) // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                else // REFLECTING BC
                {
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }

            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz)
            {
                if (param->PERIODICZ == true) // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                else // REFLECTING BC
                {
                    part->w[i] = -part->w[i];
                    part->z[i] = 2 * grd->Lz - part->z[i];
                }
            }

            if (part->z[i] < 0)
            {
                if (param->PERIODICZ == true) // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                else // REFLECTING BC
                {
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
        } // end of subcycling
    } // end of particle thread
} // end of the mover kernel



/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}
