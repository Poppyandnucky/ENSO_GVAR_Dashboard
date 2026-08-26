# Coarse Model Combination Screen

Heuristic first-pass screen only. This did not optimize or redesign the model; it ran the existing EM/KF and forecast helpers and rejected settings with obvious numerical/dynamic/economic path problems.

Grid screened:
- Countries: BRA, CHL, COL, MEX, KEN, ZAF, IND, IDN, THA, PER, PHL, EGY
- Exogenous specs: ENSO; HeatDry; HeatDryF; ENSO+HeatDry; ENSO+HeatDryF; OIL_YoY+HeatDry; OIL_YoY+HeatDryF
- MAX_EM_ITER: 10, 20, 30, 50
- UPDATE_P0: True, False
- Coefficient methods: last, avg4, avg8
- Full audit CSV: `analysis/diagnose/output/model_combo_screen_full.csv`

Important: KEEP/BORDERLINE are machine-screen labels, not final model approvals. Use them only to choose plots to inspect manually.

## Label Counts

| country | KEEP | BORDERLINE | REJECT |
| --- | --- | --- | --- |
| BRA | 0 | 0 | 169 |
| CHL | 0 | 10 | 158 |
| COL | 0 | 12 | 158 |
| EGY | 5 | 28 | 135 |
| IDN | 0 | 7 | 163 |
| IND | 23 | 16 | 130 |
| KEN | 7 | 33 | 128 |
| MEX | 24 | 1 | 147 |
| PER | 24 | 60 | 84 |
| PHL | 38 | 22 | 108 |
| THA | 0 | 13 | 155 |
| ZAF | 12 | 26 | 130 |

## BRA

No KEEP/BORDERLINE combinations under the coarse thresholds. Showing least-bad rejected settings for orientation only; inspect carefully before using.
| combination | stability | coefficients | forecast_path | counterfactual | overall | reason |
| --- | --- | --- | --- | --- | --- | --- |
| ENSO \| lag1 \| KF \| coeff=last \| EM=20 \| P0=fixed | Failed | — | — | — | REJECT | no EM result / insufficient complete data |
| ENSO \| lag1 \| KF \| coeff=last \| EM=20 \| P0=update | Failed | — | — | — | REJECT | exception: LinAlgError: SVD did not converge |
| ENSO \| lag1 \| KF \| coeff=last \| EM=30 \| P0=fixed | Failed | — | — | — | REJECT | no EM result / insufficient complete data |
| ENSO \| lag1 \| KF \| coeff=last \| EM=30 \| P0=update | Failed | — | — | — | REJECT | no EM result / insufficient complete data |
| ENSO \| lag1 \| KF \| coeff=last \| EM=50 \| P0=fixed | Failed | — | — | — | REJECT | no EM result / insufficient complete data |

Main rejection reasons:
- 93: no EM result / insufficient complete data
- 33: exception: LinAlgError: SVD did not converge
- 1: dynamic instability: rho=2695261170881075667664686731304778801216839745536.00; dynamic amplification: max ||A^k||2=81728723443057195890197496626257048123096900043860497674544282408509139552353634617483936144594020634630487067190931501140139862342999467791373131122166448077372055821349002327416558798506033152.0; coefficient explosion: max |theta|=14056541800919835557997739404380169398636869844992.0
- 1: dynamic instability: rho=530.16; dynamic amplification: max ||A^k||2=186501295996503744.0; coefficient explosion: max |theta|=3253.2
- 1: dynamic instability: rho=3436.53; dynamic amplification: max ||A^k||2=206955104587989.0; coefficient explosion: max |theta|=2937.9

## CHL

Promising combinations to inspect manually:
| combination | stability | coefficients | forecast_path | counterfactual | overall | reason |
| --- | --- | --- | --- | --- | --- | --- |
| OIL_YoY+HeatDry \| lag1 \| KF \| coeff=last \| EM=10 \| P0=fixed | Good | Borderline | Good | Acceptable | BORDERLINE | large coefficients: max \|theta\|=21.4; end-sample coefficient jump=6.6 |
| HeatDryF \| lag1 \| KF \| coeff=last \| EM=10 \| P0=fixed | Good | Borderline | Good | Acceptable | BORDERLINE | large coefficients: max \|theta\|=21.5 |
| ENSO+HeatDry \| lag1 \| KF \| coeff=last \| EM=10 \| P0=fixed | Good | Borderline | Good | Acceptable | BORDERLINE | large coefficients: max \|theta\|=22.8; end-sample coefficient jump=7.9; CI expands quickly: ratio=5.5 |
| ENSO \| lag1 \| KF \| coeff=avg4 \| EM=10 \| P0=fixed | Good | Good | Good | Acceptable | BORDERLINE | CI expands quickly: ratio=4.3 |
| ENSO \| lag1 \| KF \| coeff=avg4 \| EM=10 \| P0=update | Good | Good | Good | Acceptable | BORDERLINE | CI expands quickly: ratio=4.4 |

Main rejection reasons:
- 96: no EM result / insufficient complete data
- 21: exception: LinAlgError: SVD did not converge
- 1: coefficient explosion: max |theta|=93.1; end-sample coefficient jump=31.9
- 1: dynamic instability: rho=22.04; dynamic amplification: max ||A^k||2=1288601992.3; coefficient explosion: max |theta|=236.0
- 1: dynamic instability: rho=5.78; dynamic amplification: max ||A^k||2=1713619.3; coefficient explosion: max |theta|=222.1

## COL

Promising combinations to inspect manually:
| combination | stability | coefficients | forecast_path | counterfactual | overall | reason |
| --- | --- | --- | --- | --- | --- | --- |
| ENSO+HeatDry \| lag1 \| KF \| coeff=last \| EM=10 \| P0=update | Good | Good | Good | Acceptable | BORDERLINE | CI expands quickly: ratio=5.9 |
| ENSO+HeatDry \| lag1 \| KF \| coeff=last \| EM=10 \| P0=fixed | Good | Good | Good | Acceptable | BORDERLINE | CI expands quickly: ratio=6.0 |
| ENSO \| lag1 \| KF \| coeff=last \| EM=10 \| P0=update | Good | Good | Good | Acceptable | BORDERLINE | CI expands quickly: ratio=4.4 |
| ENSO \| lag1 \| KF \| coeff=last \| EM=10 \| P0=fixed | Good | Good | Good | Acceptable | BORDERLINE | CI expands quickly: ratio=4.4 |
| ENSO+HeatDry \| lag1 \| KF \| coeff=avg4 \| EM=10 \| P0=fixed | Good | Good | Good | Acceptable | BORDERLINE | CI expands quickly: ratio=5.9 |

Main rejection reasons:
- 105: no EM result / insufficient complete data
- 12: exception: LinAlgError: SVD did not converge
- 4: rapid CI expansion: ratio=8.9
- 3: rapid CI expansion: ratio=9.0
- 2: rapid CI expansion: ratio=18.9

## MEX

Promising combinations to inspect manually:
| combination | stability | coefficients | forecast_path | counterfactual | overall | reason |
| --- | --- | --- | --- | --- | --- | --- |
| ENSO \| lag1 \| KF \| coeff=last \| EM=10 \| P0=fixed | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO \| lag1 \| KF \| coeff=last \| EM=20 \| P0=fixed | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO \| lag1 \| KF \| coeff=last \| EM=30 \| P0=fixed | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO \| lag1 \| KF \| coeff=last \| EM=50 \| P0=fixed | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO \| lag1 \| KF \| coeff=avg4 \| EM=10 \| P0=fixed | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |

Main rejection reasons:
- 69: no EM result / insufficient complete data
- 33: exception: LinAlgError: SVD did not converge
- 4: dynamic instability: rho=13.32; dynamic amplification: max ||A^k||2=82211.5; coefficient explosion: max |theta|=30.0
- 4: dynamic instability: rho=12.09; dynamic amplification: max ||A^k||2=57657.4; coefficient explosion: max |theta|=30.0
- 4: coefficient explosion: max |theta|=30.0

## KEN

Promising combinations to inspect manually:
| combination | stability | coefficients | forecast_path | counterfactual | overall | reason |
| --- | --- | --- | --- | --- | --- | --- |
| HeatDryF \| lag1 \| KF \| coeff=last \| EM=10 \| P0=fixed | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| HeatDryF \| lag1 \| KF \| coeff=last \| EM=10 \| P0=update | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| HeatDry \| lag1 \| KF \| coeff=last \| EM=10 \| P0=update | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| HeatDryF \| lag1 \| KF \| coeff=avg4 \| EM=10 \| P0=fixed | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| HeatDryF \| lag1 \| KF \| coeff=avg8 \| EM=10 \| P0=fixed | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |

Main rejection reasons:
- 75: no EM result / insufficient complete data
- 9: exception: LinAlgError: SVD did not converge
- 1: dynamic instability: rho=1.14
- 1: dynamic instability: rho=5923101503056592951222206464.00; dynamic amplification: max ||A^k||2=169487649173749907314455980256168011217553960905616858881891565422325749629687842678949732280435801058405637238059419721327101994145371153924331292693761945439653855232.0; coefficient explosion: max |theta|=48348051024554613776649814016.0
- 1: dynamic instability: rho=77861763639064279765936521609216.00; dynamic amplification: max ||A^k||2=691251410619965931421023983369610864662077014173903676241415292702242677387570737437091517153076806882524979012929375434374067671619979662159044091733644696857126802096136779432208136778285056.0; coefficient explosion: max |theta|=398677812908915819546363780136960.0

## ZAF

Promising combinations to inspect manually:
| combination | stability | coefficients | forecast_path | counterfactual | overall | reason |
| --- | --- | --- | --- | --- | --- | --- |
| ENSO+HeatDryF \| lag1 \| KF \| coeff=avg4 \| EM=10 \| P0=update | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO+HeatDryF \| lag1 \| KF \| coeff=avg4 \| EM=20 \| P0=update | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO+HeatDryF \| lag1 \| KF \| coeff=avg4 \| EM=30 \| P0=update | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO+HeatDryF \| lag1 \| KF \| coeff=avg4 \| EM=50 \| P0=update | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO+HeatDryF \| lag1 \| KF \| coeff=last \| EM=10 \| P0=update | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |

Main rejection reasons:
- 81: no EM result / insufficient complete data
- 15: exception: LinAlgError: SVD did not converge
- 2: rapid CI expansion: ratio=9.7
- 2: rapid CI expansion: ratio=12.9
- 2: rapid CI expansion: ratio=14.8

## IND

Promising combinations to inspect manually:
| combination | stability | coefficients | forecast_path | counterfactual | overall | reason |
| --- | --- | --- | --- | --- | --- | --- |
| ENSO \| lag1 \| KF \| coeff=last \| EM=10 \| P0=fixed | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO \| lag1 \| KF \| coeff=last \| EM=10 \| P0=update | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO \| lag1 \| KF \| coeff=last \| EM=20 \| P0=update | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO \| lag1 \| KF \| coeff=last \| EM=30 \| P0=update | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO \| lag1 \| KF \| coeff=last \| EM=50 \| P0=update | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |

Main rejection reasons:
- 102: no EM result / insufficient complete data
- 9: exception: LinAlgError: SVD did not converge
- 2: dynamic instability: rho=1.09; rapid CI expansion: ratio=15.3
- 2: rapid CI expansion: ratio=8.1
- 2: dynamic instability: rho=1.09

## IDN

Promising combinations to inspect manually:
| combination | stability | coefficients | forecast_path | counterfactual | overall | reason |
| --- | --- | --- | --- | --- | --- | --- |
| HeatDry \| lag1 \| KF \| coeff=last \| EM=10 \| P0=fixed | Good | Borderline | Good | Acceptable | BORDERLINE | large coefficients: max \|theta\|=15.1; CI expands quickly: ratio=4.8 |
| HeatDryF \| lag1 \| KF \| coeff=last \| EM=10 \| P0=fixed | Good | Borderline | Good | Acceptable | BORDERLINE | large coefficients: max \|theta\|=18.9; CI expands quickly: ratio=5.8 |
| OIL_YoY+HeatDry \| lag1 \| KF \| coeff=last \| EM=10 \| P0=fixed | Good | Borderline | Good | Acceptable | BORDERLINE | large coefficients: max \|theta\|=22.1; CI expands quickly: ratio=7.5 |
| OIL_YoY+HeatDryF \| lag1 \| KF \| coeff=last \| EM=10 \| P0=fixed | Good | Borderline | Good | Acceptable | BORDERLINE | large coefficients: max \|theta\|=17.5; CI expands quickly: ratio=6.9 |
| ENSO \| lag1 \| KF \| coeff=last \| EM=10 \| P0=fixed | Good | Borderline | Good | Acceptable | BORDERLINE | large coefficients: max \|theta\|=13.8; CI expands quickly: ratio=4.6 |

Main rejection reasons:
- 99: no EM result / insufficient complete data
- 21: exception: LinAlgError: SVD did not converge
- 1: coefficient explosion: max |theta|=5092656880075145216000.0; end-sample coefficient jump=2664483599680192118784.0
- 1: dynamic instability: rho=6848524593067681336880752197453032678649672964214095872.00; dynamic amplification: max ||A^k||2=6285066659903115708153140188860164508534809044289704924605961446672128449368716749014495040762027426771195303025264375357866469949807888669117228202636568781164441355834448789384755049322560435775305310076787987018940416.0; coefficient explosion: max |theta|=10373567074054706415159730571481639919954082205169352704.0
- 1: dynamic instability: rho=17.28; dynamic amplification: max ||A^k||2=45745756.1; confidence interval explosion: max width=50222.3

## THA

Promising combinations to inspect manually:
| combination | stability | coefficients | forecast_path | counterfactual | overall | reason |
| --- | --- | --- | --- | --- | --- | --- |
| ENSO+HeatDry \| lag1 \| KF \| coeff=last \| EM=10 \| P0=update | Borderline | Good | Good | Acceptable | BORDERLINE | transient amplification: max \|\|A^k\|\|2=3.1; CI expands quickly: ratio=5.1 |
| ENSO+HeatDry \| lag1 \| KF \| coeff=last \| EM=20 \| P0=update | Borderline | Good | Good | Acceptable | BORDERLINE | transient amplification: max \|\|A^k\|\|2=3.1; CI expands quickly: ratio=5.1 |
| ENSO+HeatDry \| lag1 \| KF \| coeff=last \| EM=30 \| P0=update | Borderline | Good | Good | Acceptable | BORDERLINE | transient amplification: max \|\|A^k\|\|2=3.1; CI expands quickly: ratio=5.1 |
| ENSO+HeatDry \| lag1 \| KF \| coeff=last \| EM=50 \| P0=update | Borderline | Good | Good | Acceptable | BORDERLINE | transient amplification: max \|\|A^k\|\|2=3.1; CI expands quickly: ratio=5.1 |
| ENSO+HeatDry \| lag1 \| KF \| coeff=avg8 \| EM=10 \| P0=update | Borderline | Good | Good | Acceptable | BORDERLINE | transient amplification: max \|\|A^k\|\|2=3.1; CI expands quickly: ratio=5.1 |

Main rejection reasons:
- 90: no EM result / insufficient complete data
- 9: exception: LinAlgError: SVD did not converge
- 1: rapid CI expansion: ratio=8.8
- 1: dynamic instability: rho=23.66; dynamic amplification: max ||A^k||2=2506295.4; coefficient explosion: max |theta|=115.1
- 1: dynamic instability: rho=27.84; dynamic amplification: max ||A^k||2=4954322.5; coefficient explosion: max |theta|=160.9

## PER

Promising combinations to inspect manually:
| combination | stability | coefficients | forecast_path | counterfactual | overall | reason |
| --- | --- | --- | --- | --- | --- | --- |
| ENSO+HeatDry \| lag1 \| KF \| coeff=last \| EM=10 \| P0=update | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO+HeatDry \| lag1 \| KF \| coeff=last \| EM=20 \| P0=update | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO+HeatDry \| lag1 \| KF \| coeff=last \| EM=30 \| P0=update | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO+HeatDry \| lag1 \| KF \| coeff=last \| EM=50 \| P0=update | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO+HeatDry \| lag1 \| KF \| coeff=avg4 \| EM=10 \| P0=update | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |

Main rejection reasons:
- 54: no EM result / insufficient complete data
- 3: exception: LinAlgError: SVD did not converge
- 2: dynamic instability: rho=1.09
- 1: dynamic instability: rho=1.08
- 1: dynamic instability: rho=4607097175054268757966848.00; dynamic amplification: max ||A^k||2=63480756777897946365963144654645057677661398536767995008836753419313653571342718368974079842769358580687559716104032359433998560383924062264290705408.0; coefficient explosion: max |theta|=33891001177855758602076160.0

## PHL

Promising combinations to inspect manually:
| combination | stability | coefficients | forecast_path | counterfactual | overall | reason |
| --- | --- | --- | --- | --- | --- | --- |
| OIL_YoY+HeatDry \| lag1 \| KF \| coeff=last \| EM=10 \| P0=fixed | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| OIL_YoY+HeatDry \| lag1 \| KF \| coeff=avg4 \| EM=10 \| P0=fixed | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| OIL_YoY+HeatDry \| lag1 \| KF \| coeff=last \| EM=10 \| P0=update | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| OIL_YoY+HeatDry \| lag1 \| KF \| coeff=last \| EM=20 \| P0=update | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| OIL_YoY+HeatDry \| lag1 \| KF \| coeff=last \| EM=30 \| P0=update | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |

Main rejection reasons:
- 75: no EM result / insufficient complete data
- 3: exception: LinAlgError: SVD did not converge
- 1: dynamic instability: rho=43533466341959446614560445802740635926528.00; dynamic amplification: max ||A^k||2=29662054339127543735151795075022543090030088584432500344978976901589127220202626845186302002204357954632928575823522967857332771609460151523834103068249157636672385699856603698808302477537908735267657195279526748359920163314686505501138180112384.0; coefficient explosion: max |theta|=589816112233174912530461298535880294662144.0
- 1: dynamic instability: rho=34242227094310804237774546598655260884992.00; dynamic amplification: max ||A^k||2=16230164803564482676526040412893902607909579780634839688983867495366748730994395586535065715711291965189266234411611656927064216810364986829407993635449127103845599194455685128710878000540557183099372511465038817774509044011289517036233390292992.0; coefficient explosion: max |theta|=566296617611802103353217680297272325177344.0
- 1: dynamic instability: rho=87317364020836263507172873245949993943040.00; dynamic amplification: max ||A^k||2=1931369802079272508831352422065413183586615066210591349802093151964118236606907097217983839747148658037987148859534759043826925748472206647467220431290823849685913205311299679457510530210344554513556384145056119815923938989357594364382733353877504.0; coefficient explosion: max |theta|=589816112233174912530461298535880294662144.0

## EGY

Promising combinations to inspect manually:
| combination | stability | coefficients | forecast_path | counterfactual | overall | reason |
| --- | --- | --- | --- | --- | --- | --- |
| ENSO+HeatDry \| lag1 \| KF \| coeff=last \| EM=10 \| P0=fixed | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO+HeatDry \| lag1 \| KF \| coeff=last \| EM=20 \| P0=fixed | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO+HeatDry \| lag1 \| KF \| coeff=last \| EM=30 \| P0=fixed | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO+HeatDry \| lag1 \| KF \| coeff=last \| EM=50 \| P0=fixed | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |
| ENSO+HeatDryF \| lag1 \| KF \| coeff=last \| EM=10 \| P0=fixed | Good | Good | Good | Acceptable | KEEP | stable/coherent by coarse thresholds |

Main rejection reasons:
- 21: exception: LinAlgError: SVD did not converge
- 4: rapid CI expansion: ratio=13.2
- 4: rapid CI expansion: ratio=9.5
- 4: rapid CI expansion: ratio=11.0
- 4: rapid CI expansion: ratio=14.6
