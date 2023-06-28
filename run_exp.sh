#!/bin/bash

BINLIB=$1
PATHMAIN=$2
TUNED=$3
THRDS=$4
MATLIST=$5
CMETIS=$6


#echo $BINLIB $PATHMAIN
#load module intel
export OMP_NUM_THREADS=$THRDS
export MKL_NUM_THREADS=$THRDS


header=1
MATS="af_0_k101.mtx BenElechi1.mtx ecology2.mtx hood.mtx nd24k.mtx thermomech_dM.mtx af_shell10.mtx bmwcra_1.mtx Emilia_923.mtx Hook_1498.mtx parabolic_fem.mtx tmt_sym.mtx af_shell7.mtx bone010.mtx Fault_639.mtx ldoor.mtx PFlow_742.mtx apache2.mtx boneS10.mtx Flan_1565.mtx msdoor.mtx StocF-1465.mtx audikw_1.mtx crankseg_2.mtx G3_circuit.mtx nd12k.mtx thermal2.mtx bundle_adj.mtx pwtk.mtx m_t1.mtx x104.mtx consph.mtx shipsec5.mtx thread.mtx s3dkq4m2.mtx pdb1HYS.mtx offshore.mtx cant.mtx smt.mtx Dubcova3.mtx cfd2.mtx nasasrb.mtx ct20stif.mtx vanbody.mtx oilpan.mtx qa8fm.mtx 2cubes_sphere.mtx raefsky4.mtx msc10848.mtx denormal.mtx bcsstk36.mtx gyro.mtx olafu.mtx Pres_Poisson.mtx bundle1.mtx cbuckle.mtx fv2.mtx msc23052.mtx aft01.mtx Muu.mtx Kuu.mtx obstclae.mtx nasa2910.mtx s3rmt3m3.mtx bcsstk16.mtx Trefethen_20000.mtx bcsstk24.mtx ted_B_unscaled.mtx minsurfo.mtx"
#MATS="gyro_k.mtx Dubcova2.mtx msc23052.mtx Pres_Poisson.mtx cbuckle.mtx thermomech_dM.mtx olafu.mtx Dubcova3.mtx parabolic_fem.mtx ecology2.mtx gyro.mtx raefsky4.mtx"



if [ "$TUNED" ==  1 ]; then

#for mat in $MATS; do
while read line; do
  mat=$line
k=4
#for k in {-3,4}; do
#for lparm in {1..10}; do
#	for cparm in {1,2,3,4,5,10,20}; do
	${BINLIB}  ${PATHMAIN}/${mat} ${k} ${header} ${THRDS}
	echo ""
	if [ $header -eq 1 ]; then
     header=0
    fi
#done
#done
#done
done < ${MATLIST}
fi



if [ "$TUNED" ==  8 ]; then

#for mat in $MATS; do
while read line; do
  mat=$line
k=4
#for k in {-3,4}; do
#for lparm in {1..10}; do
#	for cparm in {1,2,3,4,5,10,20}; do
	${BINLIB}  ${PATHMAIN}/${mat} ${k} ${header} 
	echo ""
	if [ $header -eq 1 ]; then
     header=0
    fi
#done
#done
#done
done < ${MATLIST}
fi


MATSPD="Flan_1565.mtx Bump_2911.mtx Queen_4147.mtx audikw_1.mtx Serena.mtx Geo_1438.mtx Hook_1498.mtx bone010.mtx ldoor.mtx boneS10.mtx Emilia_923.mtx PFlow_742.mtx inline_1.mtx nd24k.mtx Fault_639.mtx StocF-1465.mtx bundle_adj.mtx msdoor.mtx af_shell7.mtx af_shell8.mtx af_shell4.mtx af_shell3.mtx af_3_k101.mtx af_1_k101.mtx af_4_k101.mtx af_5_k101.mtx af_0_k101.mtx af_2_k101.mtx nd12k.mtx crankseg_2.mtx BenElechi1.mtx pwtk.mtx bmwcra_1.mtx crankseg_1.mtx hood.mtx m_t1.mtx x104.mtx thermal2.mtx G3_circuit.mtx bmw7st_1.mtx nd6k.mtx consph.mtx boneS01.mtx tmt_sym.mtx ecology2.mtx apache2.mtx shipsec5.mtx thread.mtx s3dkq4m2.mtx pdb1HYS.mtx offshore.mtx cant.mtx ship_001.mtx ship_003.mtx smt.mtx s3dkt3m2.mtx parabolic_fem.mtx Dubcova3.mtx shipsec1.mtx shipsec8.mtx nd3k.mtx cfd2.mtx nasasrb.mtx ct20stif.mtx vanbody.mtx oilpan.mtx cfd1.mtx qa8fm.mtx 2cubes_sphere.mtx thermomech_dM.mtx raefsky4.mtx msc10848.mtx denormal.mtx bcsstk36.mtx msc23052.mtx Dubcova2.mtx gyro.mtx gyro_k.mtx olafu.mtx"

if [ "$TUNED" == 2 ]; then
  k=4
for mat in $MATSPD; do
#for lparm in {1..10}; do
#	for cparm in {1,2,3,4,5,10,20}; do
	$BINLIB  $PATHMAIN/$mat $k $header $THRDS
	echo ""
	if [ $header -eq 1 ]; then
     header=0
    fi
#done
#done
done
fi

MATS2="thermal2.mtx bundle_adj.mtx pwtk.mtx m_t1.mtx x104.mtx consph.mtx shipsec5.mtx thread.mtx s3dkq4m2.mtx pdb1HYS.mtx offshore.mtx cant.mtx smt.mtx Dubcova3.mtx cfd2.mtx nasasrb.mtx ct20stif.mtx vanbody.mtx oilpan.mtx qa8fm.mtx 2cubes_sphere.mtx raefsky4.mtx msc10848.mtx denormal.mtx bcsstk36.mtx gyro.mtx olafu.mtx Pres_Poisson.mtx bundle1.mtx cbuckle.mtx fv2.mtx msc23052.mtx aft01.mtx Muu.mtx Kuu.mtx obstclae.mtx nasa2910.mtx s3rmt3m3.mtx bcsstk16.mtx Trefethen_20000.mtx bcsstk24.mtx ted_B_unscaled.mtx minsurfo.mtx af_0_k101.mtx BenElechi1.mtx ecology2.mtx Emilia_923.mtx bone010.mtx Fault_639.mtx ldoor.mtx audikw_1.mtx G3_circuit.mtx nd12k.mtx"


if [ "$TUNED" == 3 ]; then
for mat in $MATS2; do
for k in {2,3,4,5,10,15,20}; do
#for lparm in {1..10}; do
#	for cparm in {1,2,3,4,5,10,20}; do
	$BINLIB  $PATHMAIN/$mat $k $header $THRDS
	echo ""
	if [ $header -eq 1 ]; then
     header=0
  fi
#done
#done
done
done
fi

#MATSP="hood.mtx bone010.mtx msdoor.mtx bundle_adj.mtx Fault_639.mtx af_shell7.mtx Kuu.mtx bcsstk24.mtx nd24k.mtx Emilia_923.mtx ldoor.mtx PFlow_742.mtx boneS10.mtx nd12k.mtx Flan_1565.mtx pwtk.mtx shipsec5.mtx smt.mtx s3rmt3m3.mtx bcsstk16.mtx Muu.mtx Trefethen_20000.mtx Dubcova3.mtx cfd2.mtx nasasrb.mtx ct20stif.mtx"

#MATSP="af_shell7 Fault_639 Flan_1565 msdoor hood BenElechi1 ldoor m_t1 x104 consph Kuu thread bmwcra_1 boneS10 s3dkq4m2 PFlow_742 Hook_1498 bone010 Emilia_923 bundle_adj"
MATSP="Flan_1565.mtx bone010.mtx Hook_1498.mtx af_shell10.mtx Emilia_923.mtx StocF-1465.mtx af_0_k101.mtx ted_B_unscaled.mtx"
if [ "$TUNED" ==  4 ]; then
for mat in $MATSP; do
k=4
	$BINLIB  "$PATHMAIN/${mat}" $k $header $THRDS
	echo ""
	if [ $header -eq 1 ]; then
     header=0
    fi
done
fi



MTT="n1024-l1.mtx n65536-l1899.mtx n4096-l1915.mtx n16384-l1893.mtx"
if [ "$TUNED" == 5 ]; then
  k=1
for mat in $MTT; do
#for lparm in {1..10}; do
#	for cparm in {1,2,3,4,5,10,20}; do
	$BINLIB  $PATHMAIN/$mat $k $header
	echo ""
	if [ $header -eq 1 ]; then
     header=0
    fi
#done
#done
done
fi



if [ "$TUNED" == 6 ]; then
  k=-2
for mat in $MATS; do
	$BINLIB  $PATHMAIN/$mat $k $header $THRDS $CMETIS
	echo ""
	if [ $header -eq 1 ]; then
     header=0
    fi
done
fi
