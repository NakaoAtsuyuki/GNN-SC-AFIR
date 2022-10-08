%link=orca
# SC-AFIR

0 1
C         -2.197882007913          1.913661198377          4.307644850497
C         -1.518473726735          0.835827577106          3.467808519709
O         -0.724783034557          0.001188977603          4.287451227578
C         -2.578759055382         -0.033672548922          2.730108191967
N         -2.859016315642         -1.199397880918          3.561674484365
C         -2.066650360965         -0.491656145602          1.355645395339
O         -1.758369389967         -1.654229552440          1.164279929511
N         -2.081051224496          0.459053979571          0.416399077328
C         -1.537334793251          0.255784547290         -0.906960236080
C         -2.651600882616          0.313647969567         -1.961202591311
C         -3.764292279851         -0.688224108685         -1.664696884859
C         -3.323526541835         -2.147248127412         -1.762772099670
C         -2.967977206035         -2.588977231688         -3.181103277714
N         -1.698363828824         -2.027512376911         -3.624207302757
C         -0.552621683969          1.385429907833         -1.160880873509
O         -0.705702937281          2.468098607329         -0.597839695900
N          0.432311693874          1.181538874701         -2.033561105754
C          1.368603378154          2.246679082623         -2.363305811192
C          1.960544474528          1.783033928749         -3.685092729587
C          2.053245927506          0.274333429739         -3.479244765167
C          0.774680902461         -0.062269120725         -2.699982793045
C          1.032394468031         -1.195449332646         -1.700865315660
O          0.678771618779         -2.343942737185         -1.913506730166
N          1.708263869427         -0.796394514980         -0.626111403294
C          1.793814580365         -1.587103783540          0.572221479758
C          3.247505043397         -1.832013897677          0.988216836413
C          3.962259453246         -0.586433455730          1.513588010024
C          4.123198936437          0.556409483499          0.493177699086
N          2.953710188088          1.390277061189          0.331478460774
C          2.573697688900          2.160547565613          1.284789593091
N          3.190137892128          2.263243888207          2.517290969269
N          1.512203443520          2.996117422089          1.117219276465
C          1.022836456123         -0.837382955361          1.649134809242
O          0.522377497884          0.242364624376          1.435418199646
O          0.962748866278         -1.451615522370          2.806563713705
H         -2.706423809178          2.625727258919          3.663902714719
H         -2.918216620116          1.459644231882          4.983668911942
H         -1.449640847114          2.439423146869          4.894561369439
H         -0.841613634351          1.299302595098          2.745368100584
H         -1.351530502022         -0.668950845734          4.625612748049
H         -3.475314367869          0.572138810171          2.574519099753
H         -2.388093200831         -1.991245126011          3.115664443331
H         -3.847706119196         -1.414540044205          3.571158937363
H         -2.153323160042          1.434604431806          0.675718074191
H         -1.054082142069         -0.719096645389         -0.919090176293
H         -2.225007120188          0.089124864083         -2.936776798903
H         -3.064392145116          1.325208479278         -1.969712229636
H         -4.580718927991         -0.515088887455         -2.368978446233
H         -4.148183709084         -0.506195225479         -0.658165515923
H         -4.146225706356         -2.773662483251         -1.411161195641
H         -2.475067948553         -2.327555047214         -1.102579592934
H         -2.963235284330         -3.691535195099         -3.203063230367
H         -3.744386506595         -2.245319547125         -3.873752401325
H         -0.958599393541         -2.354531676219         -3.002487094226
H         -1.478826595688         -2.381437959985         -4.550242347800
H          0.832348019172          3.194265430035         -2.420503585349
H          2.135307982983          2.311675064748         -1.582917917225
H          2.926824117222          2.237148692625         -3.887205260757
H          1.279412456703          2.010094965726         -4.505548507868
H          2.116369480950         -0.285162077175         -4.409010814462
H          2.924793635752          0.040606720451         -2.866932511802
H         -0.033451869146         -0.377179040708         -3.369357864881
H          1.950477873451          0.195876334349         -0.466728696319
H          1.291444577369         -2.543750458534          0.387353692014
H          3.248781467848         -2.587038309973          1.775253144198
H          3.780326335722         -2.231032341327          0.123116632002
H          3.421087603255         -0.210666891982          2.383442331592
H          4.957848205544         -0.892223563205          1.843228696588
H          4.340485400914          0.131784379508         -0.491705548699
H          4.994133703657          1.164414852658          0.784547248039
H          2.960790293444          3.080144680263          3.059680545476
H          1.063098525104          3.291684993896          1.969174084339
H          0.844807108030          2.760751568025          0.389937532053
H          0.397871284597         -0.913195193602          3.437626753627
H          4.159270385619          1.995018998068          2.559838482337
Options
DetailedOutput=ON
DownDC=15
Add Interaction
GAMMA=200.0
END
AddUniversalForce=100.0
RTemperature = 10000
MatchDecScale = 10
OrcaProc=1
OrcaMem=200
Opt=Redundant
KeepSCPath
NoFC
EQOnly
ReadBareEnergy
SC = DontCheckEntrance
SubSelectEQ=./SSE.py
SubPathsGen=./SPG.py