SELECT COUNT(*) FROM title t,movie_companies mc WHERE t.id=mc.movie_id AND mc.company_type_id=2||1333880
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_companies mc WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=112 AND mc.company_type_id=2||715
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=112||250
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=112 AND mc.company_type_id=2||715
SELECT COUNT(*) FROM title t,movie_companies mc WHERE t.id=mc.movie_id AND t.production_year>2005 AND t.production_year<2010 AND mc.company_type_id=2||211716
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_companies mc WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=113 AND mc.company_type_id=2||22
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=113 AND t.production_year>2005 AND t.production_year<2010||3
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=113 AND mc.company_type_id=2 AND t.production_year>2005 AND t.production_year<2010||9
SELECT COUNT(*) FROM title t,movie_companies mc WHERE t.id=mc.movie_id AND t.production_year>2010 AND mc.company_type_id=2||176044
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_companies mc WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=112 AND mc.company_type_id=2||715
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=112 AND t.production_year>2010||12
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=112 AND mc.company_type_id=2 AND t.production_year>2010||47
SELECT COUNT(*) FROM title t,movie_companies mc WHERE t.id=mc.movie_id AND t.production_year>2000 AND mc.company_type_id=2||619531
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_companies mc WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=113 AND mc.company_type_id=2||22
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=113 AND t.production_year>2000||6
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=113 AND mc.company_type_id=2 AND t.production_year>2000||16
SELECT COUNT(*) FROM title t,movie_companies mc WHERE t.id=mc.movie_id||2607346
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=117||148507
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=117||41818
SELECT COUNT(*) FROM movie_keyword mk,movie_companies mc,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=117||148507
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.production_year>2005||4840697
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2005||1252383
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id||232115307
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.production_year>2005||61956654
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.production_year>2010||1696239
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2010||299428
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id||232115307
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.production_year>2010||11835512
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.production_year>1990||8945392
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>1990||2741414
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id||232115307
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.production_year>1990||155153990
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2005||143429
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2005||1252383
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101||3460671
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2005 AND mi_idx.info_type_id=101||850157
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2010||43800
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2010||299428
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101||3460671
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2010 AND mi_idx.info_type_id=101||179546
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.production_year>1990||303210
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>1990||2741414
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101||3460671
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.production_year>1990 AND mi_idx.info_type_id=101||2030845
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.production_year>2005||4840697
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_type_id=2 AND t.production_year>2005||450190
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_type_id=2||19178280
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_type_id=2 AND t.production_year>2005||6244636
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.production_year>2010||1696239
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_type_id=2 AND t.production_year>2010||176044
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_type_id=2||19178280
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_type_id=2 AND t.production_year>2010||1904755
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.production_year>1990||8945392
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_type_id=2 AND t.production_year>1990||794168
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_type_id=2||19178280
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_type_id=2 AND t.production_year>1990||12551075
SELECT COUNT(*) FROM title t,movie_keyword mk WHERE t.id=mk.movie_id AND t.production_year>2010 AND mk.keyword_id=8200||11
SELECT COUNT(*) FROM title t,cast_info ci,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id=8200||1242
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>2010||5927711
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id=8200 AND t.production_year>2010||1224
SELECT COUNT(*) FROM title t,movie_keyword mk WHERE t.id=mk.movie_id AND t.production_year>2014||1054
SELECT COUNT(*) FROM title t,cast_info ci,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mk.movie_id||215711511
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>2014||3099
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND t.production_year>2014||13221
SELECT COUNT(*) FROM title t,movie_keyword mk WHERE t.id=mk.movie_id AND t.production_year>2014 AND mk.keyword_id=8200||2
SELECT COUNT(*) FROM title t,cast_info ci,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id=8200||1242
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>2014||3099
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id=8200 AND t.production_year>2014||33
SELECT COUNT(*) FROM title t,movie_keyword mk WHERE t.id=mk.movie_id AND t.production_year>2000 AND mk.keyword_id=8200||11
SELECT COUNT(*) FROM title t,cast_info ci,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id=8200||1242
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>2000||21304157
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id=8200 AND t.production_year>2000||1224
SELECT COUNT(*) FROM title t,movie_keyword mk WHERE t.id=mk.movie_id AND t.production_year>2000||1969412
SELECT COUNT(*) FROM title t,cast_info ci,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mk.movie_id||215711511
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>2000||21304157
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND t.production_year>2000||114145423
SELECT COUNT(*) FROM title t,cast_info ci WHERE t.id=ci.movie_id AND t.production_year>1980 AND t.production_year<1995||4527984
SELECT COUNT(*) FROM title t,cast_info ci WHERE t.id=ci.movie_id AND t.production_year>1980 AND t.production_year<1984||694743
SELECT COUNT(*) FROM title t,cast_info ci WHERE t.id=ci.movie_id AND t.production_year>1980 AND t.production_year<2010||21424721
SELECT COUNT(*) FROM title t,cast_info ci WHERE t.id=ci.movie_id AND ci.role_id=2||7442351
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=2||13348828
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id||2607346
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=2||13348828
SELECT COUNT(*) FROM title t,cast_info ci WHERE t.id=ci.movie_id AND ci.role_id=4||2724931
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=4||4445869
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id||2607346
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=4||4445869
SELECT COUNT(*) FROM title t,cast_info ci WHERE t.id=ci.movie_id AND ci.role_id=7||276052
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=7||794029
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id||2607346
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=7||794029
SELECT COUNT(*) FROM title t,cast_info ci WHERE t.id=ci.movie_id AND t.production_year>2005 AND t.production_year<2015 AND ci.role_id=2||3207151
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=2||13348828
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.production_year>2005 AND t.production_year<2015||828179
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=2 AND t.production_year>2005 AND t.production_year<2015||4890366
SELECT COUNT(*) FROM title t,cast_info ci WHERE t.id=ci.movie_id AND t.production_year>2007 AND t.production_year<2010 AND ci.role_id=2||881446
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=2||13348828
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.production_year>2007 AND t.production_year<2010||220921
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=2 AND t.production_year>2007 AND t.production_year<2010||1380678
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.role_id=1 AND t.production_year>2005||4857394
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.production_year>2005||828980
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=1||27309890
SELECT COUNT(*) FROM movie_companies mc,title t,cast_info ci WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.production_year>2005 AND ci.role_id=1||8715606
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.role_id=1 AND t.production_year>2010||1756997
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.production_year>2010||294397
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=1||27309890
SELECT COUNT(*) FROM movie_companies mc,title t,cast_info ci WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.production_year>2010 AND ci.role_id=1||2871880
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>1990||26738262
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.production_year>1990||1507023
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci WHERE t.id=mc.movie_id AND t.id=ci.movie_id||80225195
SELECT COUNT(*) FROM movie_companies mc,title t,cast_info ci WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.production_year>1990||56928881
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=398 AND t.production_year>1950 AND t.production_year<2000||5180
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_type_id=2 AND t.production_year>1950 AND t.production_year<2000||511428
SELECT COUNT(*) FROM title t,movie_companies mc,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_type_id=2 AND mk.keyword_id=398||14099
SELECT COUNT(*) FROM movie_companies mc,title t,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_type_id=2 AND t.production_year>1950 AND t.production_year<2000 AND mk.keyword_id=398||7150
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=398||10543
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_type_id=2||1333880
SELECT COUNT(*) FROM title t,movie_companies mc,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_type_id=2 AND mk.keyword_id=398||14099
SELECT COUNT(*) FROM movie_companies mc,title t,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_type_id=2 AND mk.keyword_id=398||14099
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>1950||3960032
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.production_year>1950||2227563
SELECT COUNT(*) FROM title t,movie_companies mc,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id||34859263
SELECT COUNT(*) FROM movie_companies mc,title t,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.production_year>1950||31334793
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=3 AND t.production_year>2005 AND t.production_year<2008||107610
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2005 AND t.production_year<2008||39821
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_type_id=2 AND t.production_year>2005 AND t.production_year<2008||97690
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=3||546095
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_type_id=2 AND mi.info_type_id=3||1527963
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_type_id=2 AND mi_idx.info_type_id=101||583864
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2005 AND t.production_year<2008 AND mi.info_type_id=3||45772
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_type_id=2 AND t.production_year>2005 AND t.production_year<2008 AND mi.info_type_id=3||118178
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_type_id=2 AND t.production_year>2005 AND t.production_year<2008 AND mi_idx.info_type_id=101||52910
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mc.company_type_id=2 AND mi.info_type_id=3 AND mi_idx.info_type_id=101||815482
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mc.company_type_id=2 AND t.production_year>2005 AND t.production_year<2008 AND mi.info_type_id=3 AND mi_idx.info_type_id=101||75436
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=105||121824
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=113||10
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id||2607346
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=113 AND mi.info_type_id=105||4
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=105||360209
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=113||120
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=113 AND mi.info_type_id=105||4
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=105||360209
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=113||120
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=105 AND mi_idx.info_type_id=113||72
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=105 AND mi_idx.info_type_id=113||72
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=3 AND t.production_year>2000 AND t.production_year<2010||435574
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2000 AND t.production_year<2010||157103
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_type_id=2 AND t.production_year>2000 AND t.production_year<2010||381057
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=3||546095
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_type_id=2 AND mi.info_type_id=3||1527963
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_type_id=2 AND mi_idx.info_type_id=101||583864
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2000 AND t.production_year<2010 AND mi.info_type_id=3||185473
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_type_id=2 AND t.production_year>2000 AND t.production_year<2010 AND mi.info_type_id=3||474089
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_type_id=2 AND t.production_year>2000 AND t.production_year<2010 AND mi_idx.info_type_id=101||207436
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mc.company_type_id=2 AND mi.info_type_id=3 AND mi_idx.info_type_id=101||815482
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mc.company_type_id=2 AND t.production_year>2000 AND t.production_year<2010 AND mi.info_type_id=3 AND mi_idx.info_type_id=101||305655
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.kind_id=1||1083597
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_type_id=2 AND t.kind_id=1||640620
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.kind_id=1||209880
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_type_id=2 AND mi.info_type_id=16||3241660
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=16||1118379
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_companies mc WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=101 AND mc.company_type_id=2||583864
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_type_id=2 AND t.kind_id=1 AND mi.info_type_id=16||2250196
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND t.kind_id=1 AND mi.info_type_id=16||724731
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_companies mc WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=101 AND t.kind_id=1 AND mc.company_type_id=2||302610
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_info mi,movie_companies mc WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=16 AND mc.company_type_id=2||2504235
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_info mi,movie_companies mc WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=101 AND t.kind_id=1 AND mi.info_type_id=16 AND mc.company_type_id=2||1919495
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=8 AND t.production_year>2010 AND t.kind_id=1||108646
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2010 AND t.kind_id=1||17526
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2010 AND t.kind_id=1||164339
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=8||424768
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8||5074855
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101||3460671
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2010 AND t.kind_id=1 AND mi.info_type_id=8||20716
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.production_year>2010 AND t.kind_id=1 AND mi.info_type_id=8||209492
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2010 AND t.kind_id=1 AND mi_idx.info_type_id=101||110975
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=8 AND mi_idx.info_type_id=101||4167739
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2010 AND t.kind_id=1 AND mi.info_type_id=8 AND mi_idx.info_type_id=101||150780
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=8 AND t.kind_id=1||705350
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.kind_id=1||209880
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=1||2886675
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=8||424768
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8||5074855
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101||3460671
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND t.kind_id=1 AND mi.info_type_id=8||238948
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.kind_id=1 AND mi.info_type_id=8||3643079
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=1 AND mi_idx.info_type_id=101||2503305
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=8 AND mi_idx.info_type_id=101||4167739
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=1 AND mi.info_type_id=8 AND mi_idx.info_type_id=101||3243247
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=8 AND t.production_year>2005||454945
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2005||143429
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2005||1252383
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=8||424768
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8||5074855
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101||3460671
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2005 AND mi.info_type_id=8||125453
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.production_year>2005 AND mi.info_type_id=8||1383624
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2005 AND mi_idx.info_type_id=101||850157
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=8 AND mi_idx.info_type_id=101||4167739
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2005 AND mi.info_type_id=8 AND mi_idx.info_type_id=101||1043368
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.production_year>2000||1796049
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.production_year>2000||1169320
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2000||1969412
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16||10901928
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16||36410096
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id||34859263
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.production_year>2000 AND mi.info_type_id=16||6619451
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.production_year>2000 AND mi.info_type_id=16||21792864
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.production_year>2000||16654860
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id=16||753806188
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND t.production_year>2000 AND mi.info_type_id=16||512565359
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.production_year>2005 AND t.production_year<2010||669110
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.production_year>2005 AND t.production_year<2010||420076
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2005 AND t.production_year<2010||776759
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16||10901928
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16||36410096
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id||34859263
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.production_year>2005 AND t.production_year<2010 AND mi.info_type_id=16||2557161
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.production_year>2005 AND t.production_year<2010 AND mi.info_type_id=16||8541179
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.production_year>2005 AND t.production_year<2010||6669232
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id=16||753806188
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND t.production_year>2005 AND t.production_year<2010 AND mi.info_type_id=16||206772699
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.production_year>1990||2217197
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.production_year>1990||1507023
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>1990||2741414
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16||10901928
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16||36410096
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id||34859263
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.production_year>1990 AND mi.info_type_id=16||8191413
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.production_year>1990 AND mi.info_type_id=16||27650712
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.production_year>1990||22576296
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id=16||753806188
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND t.production_year>1990 AND mi.info_type_id=16||625291253
SELECT COUNT(*) FROM title t,cast_info ci WHERE t.id=ci.movie_id||36198131
SELECT COUNT(*) FROM title t,movie_keyword mk,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id=117||1038078
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci WHERE t.id=mc.movie_id AND t.id=ci.movie_id||80225195
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=117||41818
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id||2607346
SELECT COUNT(*) FROM title t,movie_companies mc,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id=117||148507
SELECT COUNT(*) FROM movie_keyword mk,cast_info ci,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id=117||1038078
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id||80225195
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id=117||7796133
SELECT COUNT(*) FROM movie_companies mc,title t,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id=117||148507
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id=117||7796133
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=105||121824
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100||459562
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id||36198131
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=100 AND mi.info_type_id=105||45418
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=105||2766349
SELECT COUNT(*) FROM title t,cast_info ci,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100||13614507
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=100 AND mi.info_type_id=105||45418
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=105||2766349
SELECT COUNT(*) FROM cast_info ci,title t,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100||13614507
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=105 AND mi_idx.info_type_id=100||1830786
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=105 AND mi_idx.info_type_id=100||1830786
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=3 AND t.production_year>2008 AND t.production_year<2014||456984
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2008 AND t.production_year<2014||84094
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>2008 AND t.production_year<2014||10370922
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=3||546095
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3||26034953
SELECT COUNT(*) FROM title t,cast_info ci,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101||13614507
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2008 AND t.production_year<2014 AND mi.info_type_id=3||96354
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year>2008 AND t.production_year<2014 AND mi.info_type_id=3||7289665
SELECT COUNT(*) FROM cast_info ci,title t,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2008 AND t.production_year<2014 AND mi_idx.info_type_id=101||2837998
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=101||16460162
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2008 AND t.production_year<2014 AND mi.info_type_id=3 AND mi_idx.info_type_id=101||2935857
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=3||1533668
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100||459562
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id||36198131
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=100 AND mi.info_type_id=3||546095
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3||26034953
SELECT COUNT(*) FROM title t,cast_info ci,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100||13614507
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=100 AND mi.info_type_id=3||546095
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3||26034953
SELECT COUNT(*) FROM cast_info ci,title t,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100||13614507
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100||16460162
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100||16460162
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.production_year>2005 AND t.production_year<2009||494730
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.production_year>2005 AND t.production_year<2009||304819
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.role_id=2 AND t.production_year>2005 AND t.production_year<2009||1190249
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16||10901928
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.role_id=2 AND mi.info_type_id=16||14791118
SELECT COUNT(*) FROM title t,cast_info ci,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=2||13348828
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.production_year>2005 AND t.production_year<2009 AND mi.info_type_id=16||1816266
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.role_id=2 AND t.production_year>2005 AND t.production_year<2009 AND mi.info_type_id=16||2644709
SELECT COUNT(*) FROM cast_info ci,title t,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND t.production_year>2005 AND t.production_year<2009||1836426
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND mi.info_type_id=16||132301269
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND t.production_year>2005 AND t.production_year<2009 AND mi.info_type_id=16||24671503
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16||3033192
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id||2607346
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.role_id=2||7442351
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16||10901928
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.role_id=2 AND mi.info_type_id=16||14791118
SELECT COUNT(*) FROM title t,cast_info ci,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=2||13348828
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16||10901928
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.role_id=2 AND mi.info_type_id=16||14791118
SELECT COUNT(*) FROM cast_info ci,title t,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=2||13348828
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND mi.info_type_id=16||132301269
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND mi.info_type_id=16||132301269
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.production_year>2000||1796049
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.production_year>2000||1169320
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.role_id=2 AND t.production_year>2000||4395831
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16||10901928
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.role_id=2 AND mi.info_type_id=16||14791118
SELECT COUNT(*) FROM title t,cast_info ci,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=2||13348828
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.production_year>2000 AND mi.info_type_id=16||6619451
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.role_id=2 AND t.production_year>2000 AND mi.info_type_id=16||9265596
SELECT COUNT(*) FROM cast_info ci,title t,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND t.production_year>2000||6987414
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND mi.info_type_id=16||132301269
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND t.production_year>2000 AND mi.info_type_id=16||90637406
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>1950 AND t.kind_id=1||8852160
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>1950 AND t.kind_id=1||2343162
SELECT COUNT(*) FROM title t,movie_keyword mk,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id||215711511
SELECT COUNT(*) FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>1950 AND t.kind_id=1||151178411
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>2000 AND t.kind_id=1||4911602
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2000 AND t.kind_id=1||1053171
SELECT COUNT(*) FROM title t,movie_keyword mk,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id||215711511
SELECT COUNT(*) FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2000 AND t.kind_id=1||84216099
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=398 AND t.production_year>1950 AND t.production_year<2000||5180
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_type_id=2 AND t.production_year>1950 AND t.production_year<2000||511428
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.production_year>1950 AND t.production_year<2000||5438873
SELECT COUNT(*) FROM title t,movie_companies mc,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_type_id=2 AND mk.keyword_id=398||14099
SELECT COUNT(*) FROM title t,movie_info mi,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id=398||322881
SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_type_id=2||19178280
SELECT COUNT(*) FROM movie_companies mc,title t,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_type_id=2 AND t.production_year>1950 AND t.production_year<2000 AND mk.keyword_id=398||7150
SELECT COUNT(*) FROM movie_info mi,title t,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year>1950 AND t.production_year<2000 AND mk.keyword_id=398||166764
SELECT COUNT(*) FROM movie_info mi,title t,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.production_year>1950 AND t.production_year<2000 AND mc.company_type_id=2||7151252
SELECT COUNT(*) FROM title t,movie_info mi,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2||767549
SELECT COUNT(*) FROM movie_info mi,title t,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND t.production_year>1950 AND t.production_year<2000 AND mk.keyword_id=398 AND mc.company_type_id=2||328043
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=398 AND t.production_year>2000 AND t.production_year<2010||2699
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_type_id=2 AND t.production_year>2000 AND t.production_year<2010||381057
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.production_year>2000 AND t.production_year<2010||4487176
SELECT COUNT(*) FROM title t,movie_companies mc,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_type_id=2 AND mk.keyword_id=398||14099
SELECT COUNT(*) FROM title t,movie_info mi,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id=398||322881
SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_type_id=2||19178280
SELECT COUNT(*) FROM movie_companies mc,title t,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_type_id=2 AND t.production_year>2000 AND t.production_year<2010 AND mk.keyword_id=398||3411
SELECT COUNT(*) FROM movie_info mi,title t,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year>2000 AND t.production_year<2010 AND mk.keyword_id=398||94482
SELECT COUNT(*) FROM movie_info mi,title t,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.production_year>2000 AND t.production_year<2010 AND mc.company_type_id=2||6774760
SELECT COUNT(*) FROM title t,movie_info mi,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2||767549
SELECT COUNT(*) FROM movie_info mi,title t,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND t.production_year>2000 AND t.production_year<2010 AND mk.keyword_id=398 AND mc.company_type_id=2||304269
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=398 AND t.production_year>1950 AND t.production_year<2010||8208
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_type_id=2 AND t.production_year>1950 AND t.production_year<2010||916656
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.production_year>1950 AND t.production_year<2010||10217957
SELECT COUNT(*) FROM title t,movie_companies mc,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_type_id=2 AND mk.keyword_id=398||14099
SELECT COUNT(*) FROM title t,movie_info mi,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id=398||322881
SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_type_id=2||19178280
SELECT COUNT(*) FROM movie_companies mc,title t,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_type_id=2 AND t.production_year>1950 AND t.production_year<2010 AND mk.keyword_id=398||10933
SELECT COUNT(*) FROM movie_info mi,title t,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year>1950 AND t.production_year<2010 AND mk.keyword_id=398||268774
SELECT COUNT(*) FROM movie_info mi,title t,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.production_year>1950 AND t.production_year<2010 AND mc.company_type_id=2||14395031
SELECT COUNT(*) FROM title t,movie_info mi,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2||767549
SELECT COUNT(*) FROM movie_info mi,title t,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND t.production_year>1950 AND t.production_year<2010 AND mk.keyword_id=398 AND mc.company_type_id=2||649994
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=8 AND t.production_year>2008||296042
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2008||84095
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2008||683650
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.production_year>2008||524161
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=8||424768
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8||5074855
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8||2564666
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101||3460671
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101||1354134
SELECT COUNT(*) FROM title t,movie_companies mc,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id||34859263
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2008 AND mi.info_type_id=8||71247
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.production_year>2008 AND mi.info_type_id=8||731450
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.production_year>2008 AND mi.info_type_id=8||494012
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2008 AND mi_idx.info_type_id=101||447345
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2008 AND mi_idx.info_type_id=101||248443
SELECT COUNT(*) FROM movie_companies mc,title t,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.production_year>2008||5552714
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=8 AND mi_idx.info_type_id=101||4167739
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=8 AND mi_idx.info_type_id=101||1581949
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=8||50641056
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101||33223861
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2008 AND mi.info_type_id=8 AND mi_idx.info_type_id=101||531875
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2008 AND mi.info_type_id=8 AND mi_idx.info_type_id=101||288897
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year>2008 AND mi.info_type_id=8||8558482
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>2008 AND mi_idx.info_type_id=101||5264774
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi.info_type_id=8 AND mi_idx.info_type_id=101||49073088
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>2008 AND mi.info_type_id=8 AND mi_idx.info_type_id=101||8274618
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=8 AND t.production_year>2009||233477
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2009||64002
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2009||475624
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.production_year>2009||408904
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=8||424768
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8||5074855
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8||2564666
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101||3460671
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101||1354134
SELECT COUNT(*) FROM title t,movie_companies mc,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id||34859263
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2009 AND mi.info_type_id=8||52984
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.production_year>2009 AND mi.info_type_id=8||487939
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.production_year>2009 AND mi.info_type_id=8||374317
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2009 AND mi_idx.info_type_id=101||305439
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2009 AND mi_idx.info_type_id=101||183720
SELECT COUNT(*) FROM movie_companies mc,title t,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.production_year>2009||3654658
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=8 AND mi_idx.info_type_id=101||4167739
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=8 AND mi_idx.info_type_id=101||1581949
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=8||50641056
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101||33223861
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2009 AND mi.info_type_id=8 AND mi_idx.info_type_id=101||346669
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2009 AND mi.info_type_id=8 AND mi_idx.info_type_id=101||207944
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year>2009 AND mi.info_type_id=8||5280393
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>2009 AND mi_idx.info_type_id=101||3431494
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi.info_type_id=8 AND mi_idx.info_type_id=101||49073088
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>2009 AND mi.info_type_id=8 AND mi_idx.info_type_id=101||5060063
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.production_year>2010||435976
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.production_year>2010||294397
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.role_id=2 AND t.production_year>2010||1126191
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2010||299428
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16||10901928
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.role_id=2 AND mi.info_type_id=16||14791118
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16||36410096
SELECT COUNT(*) FROM title t,cast_info ci,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=2||13348828
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id||34859263
SELECT COUNT(*) FROM title t,movie_keyword mk,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.role_id=2||35799253
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.production_year>2010 AND mi.info_type_id=16||1257875
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.role_id=2 AND t.production_year>2010 AND mi.info_type_id=16||1982743
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.production_year>2010 AND mi.info_type_id=16||2742116
SELECT COUNT(*) FROM cast_info ci,title t,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND t.production_year>2010||1636053
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.production_year>2010||2116410
SELECT COUNT(*) FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2010 AND ci.role_id=2||2583694
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND mi.info_type_id=16||132301269
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id=16||753806188
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id=16 AND ci.role_id=2||540408036
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=2||434295845
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND t.production_year>2010 AND mi.info_type_id=16||17669815
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND t.production_year>2010 AND mi.info_type_id=16||67596967
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=ci.movie_id AND t.production_year>2010 AND mi.info_type_id=16 AND ci.role_id=2||50178676
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.production_year>2010 AND ci.role_id=2||32903354
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND mi.info_type_id=16 AND ci.role_id=2||12883046375
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.production_year>2010 AND mi.info_type_id=16 AND ci.role_id=2||1353168516
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.production_year>2010||435976
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=22956 AND t.production_year>2010||37
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.role_id=2 AND t.production_year>2010||1126191
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2010||299428
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id=22956 AND mi.info_type_id=16||1680
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.role_id=2 AND mi.info_type_id=16||14791118
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16||36410096
SELECT COUNT(*) FROM title t,cast_info ci,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND mc.company_id=22956||339
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id=22956||2756
SELECT COUNT(*) FROM title t,movie_keyword mk,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.role_id=2||35799253
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id=22956 AND t.production_year>2010 AND mi.info_type_id=16||430
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.role_id=2 AND t.production_year>2010 AND mi.info_type_id=16||1982743
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.production_year>2010 AND mi.info_type_id=16||2742116
SELECT COUNT(*) FROM cast_info ci,title t,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND t.production_year>2010 AND mc.company_id=22956||86
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.production_year>2010 AND mc.company_id=22956||449
SELECT COUNT(*) FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2010 AND ci.role_id=2||2583694
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND mc.company_id=22956||13040
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id=16 AND mc.company_id=22956||133709
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id=16 AND ci.role_id=2||540408036
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id=22956 AND ci.role_id=2||20207
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND t.production_year>2010 AND mi.info_type_id=16 AND mc.company_id=22956||2967
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND t.production_year>2010 AND mi.info_type_id=16 AND mc.company_id=22956||21091
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=ci.movie_id AND t.production_year>2010 AND mi.info_type_id=16 AND ci.role_id=2||50178676
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.production_year>2010 AND mc.company_id=22956 AND ci.role_id=2||3340
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND mi.info_type_id=16 AND mc.company_id=22956 AND ci.role_id=2||1135407
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.production_year>2010 AND mi.info_type_id=16 AND mc.company_id=22956 AND ci.role_id=2||191935
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.production_year>2000||1796049
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.production_year>2000||1169320
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.role_id=2 AND t.production_year>2000||4395831
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2000||1969412
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16||10901928
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.role_id=2 AND mi.info_type_id=16||14791118
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16||36410096
SELECT COUNT(*) FROM title t,cast_info ci,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=2||13348828
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id||34859263
SELECT COUNT(*) FROM title t,movie_keyword mk,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.role_id=2||35799253
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.production_year>2000 AND mi.info_type_id=16||6619451
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.role_id=2 AND t.production_year>2000 AND mi.info_type_id=16||9265596
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.production_year>2000 AND mi.info_type_id=16||21792864
SELECT COUNT(*) FROM cast_info ci,title t,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND t.production_year>2000||6987414
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.production_year>2000||16654860
SELECT COUNT(*) FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2000 AND ci.role_id=2||18049664
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND mi.info_type_id=16||132301269
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id=16||753806188
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id=16 AND ci.role_id=2||540408036
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=2||434295845
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND t.production_year>2000 AND mi.info_type_id=16||90637406
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND t.production_year>2000 AND mi.info_type_id=16||512565359
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=ci.movie_id AND t.production_year>2000 AND mi.info_type_id=16 AND ci.role_id=2||365166349
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.production_year>2000 AND ci.role_id=2||251877937
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND mi.info_type_id=16 AND ci.role_id=2||12883046375
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.production_year>2000 AND mi.info_type_id=16 AND ci.role_id=2||9537609641
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=3||1533668
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100||459562
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id||36198131
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id||4522201
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=100 AND mi.info_type_id=3||546095
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3||26034953
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3||8853937
SELECT COUNT(*) FROM title t,cast_info ci,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100||13614507
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100||3460671
SELECT COUNT(*) FROM title t,movie_keyword mk,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id||215711511
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=100 AND mi.info_type_id=3||546095
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3||26034953
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3||8853937
SELECT COUNT(*) FROM cast_info ci,title t,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100||13614507
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100||3460671
SELECT COUNT(*) FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id||215711511
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100||16460162
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100||7566897
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id=3||510911332
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info_idx mi_idx,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id=100||200499873
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100||16460162
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100||7566897
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id=3||510911332
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info_idx mi_idx,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id=100||200499873
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_info_idx mi_idx,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100||492932174
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_info_idx mi_idx,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100||492932174
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=3 AND t.production_year>2010||276591
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.production_year>2010||43800
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>2010||5927711
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2010||299428
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=100 AND mi.info_type_id=3||546095
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3||26034953
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3||8853937
SELECT COUNT(*) FROM title t,cast_info ci,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100||13614507
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100||3460671
SELECT COUNT(*) FROM title t,movie_keyword mk,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id||215711511
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=100 AND t.production_year>2010 AND mi.info_type_id=3||46639
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year>2010 AND mi.info_type_id=3||4308398
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.production_year>2010 AND mi.info_type_id=3||463848
SELECT COUNT(*) FROM cast_info ci,title t,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2010 AND mi_idx.info_type_id=100||1489436
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2010 AND mi_idx.info_type_id=100||179546
SELECT COUNT(*) FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2010||17276078
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100||16460162
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100||7566897
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id=3||510911332
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info_idx mi_idx,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id=100||200499873
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2010 AND mi.info_type_id=3 AND mi_idx.info_type_id=100||1436118
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2010 AND mi.info_type_id=3 AND mi_idx.info_type_id=100||306404
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=ci.movie_id AND t.production_year>2010 AND mi.info_type_id=3||35754660
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info_idx mi_idx,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.production_year>2010 AND mi_idx.info_type_id=100||14790028
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_info_idx mi_idx,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100||492932174
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_info_idx mi_idx,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.production_year>2010 AND mi.info_type_id=3 AND mi_idx.info_type_id=100||32355195
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>2000 AND t.kind_id=1||4911602
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2000 AND t.kind_id=1||1053171
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2000 AND t.kind_id=1||86572
SELECT COUNT(*) FROM title t,movie_keyword mk,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id||215711511
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,cast_info ci WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id=101||13614507
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101||3460671
SELECT COUNT(*) FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2000 AND t.kind_id=1||84216099
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,cast_info ci WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2000 AND t.kind_id=1||2693643
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_keyword mk WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2000 AND t.kind_id=1||897347
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,cast_info ci,movie_keyword mk WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101||200499873
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,cast_info ci,movie_keyword mk WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2000 AND t.kind_id=1||81494089
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>2005 AND t.kind_id=1||3785488
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2005 AND t.kind_id=1||682355
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2005 AND t.kind_id=1||58024
SELECT COUNT(*) FROM title t,movie_keyword mk,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id||215711511
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,cast_info ci WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id=101||13614507
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101||3460671
SELECT COUNT(*) FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2005 AND t.kind_id=1||53764996
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,cast_info ci WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2005 AND t.kind_id=1||1839551
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_keyword mk WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2005 AND t.kind_id=1||547364
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,cast_info ci,movie_keyword mk WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101||200499873
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,cast_info ci,movie_keyword mk WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2005 AND t.kind_id=1||51282177
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=398 AND t.production_year=1998||271
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_type_id=2 AND t.production_year=1998||20439
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.production_year=1998||246860
SELECT COUNT(*) FROM title t,movie_companies mc,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_type_id=2 AND mk.keyword_id=398||14099
SELECT COUNT(*) FROM title t,movie_info mi,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id=398||322881
SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_type_id=2||19178280
SELECT COUNT(*) FROM movie_companies mc,title t,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_type_id=2 AND t.production_year=1998 AND mk.keyword_id=398||372
SELECT COUNT(*) FROM movie_info mi,title t,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year=1998 AND mk.keyword_id=398||8444
SELECT COUNT(*) FROM movie_info mi,title t,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.production_year=1998 AND mc.company_type_id=2||364270
SELECT COUNT(*) FROM title t,movie_info mi,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2||767549
SELECT COUNT(*) FROM movie_info mi,title t,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND t.production_year=1998 AND mk.keyword_id=398 AND mc.company_type_id=2||18793
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=8 AND t.production_year>2000||641099
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2000||221105
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2000||1969412
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.production_year>2000||1169320
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=8||424768
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8||5074855
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8||2564666
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101||3460671
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101||1354134
SELECT COUNT(*) FROM title t,movie_companies mc,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id||34859263
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2000 AND mi.info_type_id=8||198789
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.production_year>2000 AND mi.info_type_id=8||2283023
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.production_year>2000 AND mi.info_type_id=8||1194720
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2000 AND mi_idx.info_type_id=101||1402286
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2000 AND mi_idx.info_type_id=101||645645
SELECT COUNT(*) FROM movie_companies mc,title t,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.production_year>2000||16654860
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=8 AND mi_idx.info_type_id=101||4167739
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=8 AND mi_idx.info_type_id=101||1581949
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=8||50641056
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101||33223861
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2000 AND mi.info_type_id=8 AND mi_idx.info_type_id=101||1800605
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2000 AND mi.info_type_id=8 AND mi_idx.info_type_id=101||783905
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year>2000 AND mi.info_type_id=8||27357449
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>2000 AND mi_idx.info_type_id=101||15998483
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi.info_type_id=8 AND mi_idx.info_type_id=101||49073088
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>2000 AND mi.info_type_id=8 AND mi_idx.info_type_id=101||26715485
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=8 AND t.production_year>2005||454945
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2005||143429
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2005||1252383
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.production_year>2005||828980
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=8||424768
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8||5074855
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8||2564666
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101||3460671
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101||1354134
SELECT COUNT(*) FROM title t,movie_companies mc,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id||34859263
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101 AND t.production_year>2005 AND mi.info_type_id=8||125453
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.production_year>2005 AND mi.info_type_id=8||1383624
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.production_year>2005 AND mi.info_type_id=8||813294
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2005 AND mi_idx.info_type_id=101||850157
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2005 AND mi_idx.info_type_id=101||427402
SELECT COUNT(*) FROM movie_companies mc,title t,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.production_year>2005||10323890
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=8 AND mi_idx.info_type_id=101||4167739
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=8 AND mi_idx.info_type_id=101||1581949
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=8||50641056
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101||33223861
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2005 AND mi.info_type_id=8 AND mi_idx.info_type_id=101||1043368
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2005 AND mi.info_type_id=8 AND mi_idx.info_type_id=101||508150
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year>2005 AND mi.info_type_id=8||16261850
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>2005 AND mi_idx.info_type_id=101||9859330
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi.info_type_id=8 AND mi_idx.info_type_id=101||49073088
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>2005 AND mi.info_type_id=8 AND mi_idx.info_type_id=101||15811172
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.production_year>2000 AND t.production_year<2010||1183646
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.production_year>2000 AND t.production_year<2010||760416
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.role_id=2 AND t.production_year>2000 AND t.production_year<2010||2828089
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=7084 AND t.production_year>2000 AND t.production_year<2010||199
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16||10901928
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.role_id=2 AND mi.info_type_id=16||14791118
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id=7084 AND mi.info_type_id=16||4801
SELECT COUNT(*) FROM title t,cast_info ci,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=2||13348828
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=7084||3218
SELECT COUNT(*) FROM title t,movie_keyword mk,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id=7084 AND ci.role_id=2||1821
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.production_year>2000 AND t.production_year<2010 AND mi.info_type_id=16||4732082
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.role_id=2 AND t.production_year>2000 AND t.production_year<2010 AND mi.info_type_id=16||6390179
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id=7084 AND t.production_year>2000 AND t.production_year<2010 AND mi.info_type_id=16||3456
SELECT COUNT(*) FROM cast_info ci,title t,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND t.production_year>2000 AND t.production_year<2010||4671122
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=7084 AND t.production_year>2000 AND t.production_year<2010||2178
SELECT COUNT(*) FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id=7084 AND t.production_year>2000 AND t.production_year<2010 AND ci.role_id=2||1254
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND mi.info_type_id=16||132301269
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND mk.keyword_id=7084 AND mi.info_type_id=16||114260
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=ci.movie_id AND mk.keyword_id=7084 AND mi.info_type_id=16 AND ci.role_id=2||57861
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND mk.keyword_id=7084 AND ci.role_id=2||32299
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND t.production_year>2000 AND t.production_year<2010 AND mi.info_type_id=16||64286754
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND mk.keyword_id=7084 AND t.production_year>2000 AND t.production_year<2010 AND mi.info_type_id=16||83888
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=ci.movie_id AND mk.keyword_id=7084 AND t.production_year>2000 AND t.production_year<2010 AND mi.info_type_id=16 AND ci.role_id=2||41636
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND mk.keyword_id=7084 AND t.production_year>2000 AND t.production_year<2010 AND ci.role_id=2||23853
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND mk.keyword_id=7084 AND mi.info_type_id=16 AND ci.role_id=2||1499619
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND mk.keyword_id=7084 AND t.production_year>2000 AND t.production_year<2010 AND mi.info_type_id=16 AND ci.role_id=2||1067496
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.production_year>2000 AND t.production_year<2005||380353
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.production_year>2000 AND t.production_year<2005||252907
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.role_id=2 AND t.production_year>2000 AND t.production_year<2005||887119
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=7084 AND t.production_year>2000 AND t.production_year<2005||100
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16||10901928
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.role_id=2 AND mi.info_type_id=16||14791118
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id=7084 AND mi.info_type_id=16||4801
SELECT COUNT(*) FROM title t,cast_info ci,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=2||13348828
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=7084||3218
SELECT COUNT(*) FROM title t,movie_keyword mk,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id=7084 AND ci.role_id=2||1821
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info mi WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.production_year>2000 AND t.production_year<2005 AND mi.info_type_id=16||1672518
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.role_id=2 AND t.production_year>2000 AND t.production_year<2005 AND mi.info_type_id=16||2053786
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id=7084 AND t.production_year>2000 AND t.production_year<2005 AND mi.info_type_id=16||1084
SELECT COUNT(*) FROM cast_info ci,title t,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND t.production_year>2000 AND t.production_year<2005||1570351
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=7084 AND t.production_year>2000 AND t.production_year<2005||777
SELECT COUNT(*) FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id=7084 AND t.production_year>2000 AND t.production_year<2005 AND ci.role_id=2||467
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND mi.info_type_id=16||132301269
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND mk.keyword_id=7084 AND mi.info_type_id=16||114260
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=ci.movie_id AND mk.keyword_id=7084 AND mi.info_type_id=16 AND ci.role_id=2||57861
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND mk.keyword_id=7084 AND ci.role_id=2||32299
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND ci.role_id=2 AND t.production_year>2000 AND t.production_year<2005 AND mi.info_type_id=16||22137133
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND mk.keyword_id=7084 AND t.production_year>2000 AND t.production_year<2005 AND mi.info_type_id=16||22926
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=ci.movie_id AND mk.keyword_id=7084 AND t.production_year>2000 AND t.production_year<2005 AND mi.info_type_id=16 AND ci.role_id=2||11235
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND mk.keyword_id=7084 AND t.production_year>2000 AND t.production_year<2005 AND ci.role_id=2||7143
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND mk.keyword_id=7084 AND mi.info_type_id=16 AND ci.role_id=2||1499619
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_companies mc,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND mk.keyword_id=7084 AND t.production_year>2000 AND t.production_year<2005 AND mi.info_type_id=16 AND ci.role_id=2||268172
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=3 AND t.production_year>2000||815079
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.production_year>2000||221105
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>2000||21304157
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.production_year>2000||1969412
SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=100 AND mi.info_type_id=3||546095
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3||26034953
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3||8853937
SELECT COUNT(*) FROM title t,cast_info ci,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100||13614507
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100||3460671
SELECT COUNT(*) FROM title t,movie_keyword mk,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id||215711511
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t,movie_info mi WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=100 AND t.production_year>2000 AND mi.info_type_id=3||257589
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year>2000 AND mi.info_type_id=3||14391234
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.production_year>2000 AND mi.info_type_id=3||3564030
SELECT COUNT(*) FROM cast_info ci,title t,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2000 AND mi_idx.info_type_id=100||7090266
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2000 AND mi_idx.info_type_id=100||1402286
SELECT COUNT(*) FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2000||114145423
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100||16460162
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100||7566897
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id=3||510911332
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info_idx mi_idx,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id=100||200499873
SELECT COUNT(*) FROM cast_info ci,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2000 AND mi.info_type_id=3 AND mi_idx.info_type_id=100||8005030
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2000 AND mi.info_type_id=3 AND mi_idx.info_type_id=100||2912009
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=ci.movie_id AND t.production_year>2000 AND mi.info_type_id=3||268232828
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info_idx mi_idx,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.production_year>2000 AND mi_idx.info_type_id=100||104953995
SELECT COUNT(*) FROM title t,movie_keyword mk,movie_info mi,movie_info_idx mi_idx,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100||492932174
SELECT COUNT(*) FROM movie_keyword mk,title t,movie_info mi,movie_info_idx mi_idx,cast_info ci WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.production_year>2000 AND mi.info_type_id=3 AND mi_idx.info_type_id=100||258380423
