from DataPreparation.graph_representation import SchemaGraph, Table


def gen_power_schema(csv_path):
    schema = SchemaGraph()

    schema.add_table(Table('power', attributes=['Global_active_power', 'Global_reactive_power', 'Voltage',
                                                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
                                                'Sub_metering_3'],
                           irrelevant_attributes=[],
                           no_compression=[],
                           csv_file_location=csv_path.format('power'),
                           table_size=2075259))

    return schema


def gen_forest_schema(csv_path):
    schema = SchemaGraph()

    schema.add_table(Table('forest', attributes=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                                                 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                                                 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                                                 'Horizontal_Distance_To_Fire_Points', ],
                           irrelevant_attributes=[],
                           no_compression=[],
                           csv_file_location=csv_path.format('forest'),
                           table_size=581012))

    return schema


def gen_census_schema(csv_path):
    schema = SchemaGraph()

    schema.add_table(Table('census', attributes=['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
                                                 'occupation', 'relationship', 'race', 'sex', 'capital_gain',
                                                 'capital_loss', 'hours_per_week', 'native_country', 'income_level'],
                           irrelevant_attributes=[],
                           no_compression=[],
                           csv_file_location=csv_path.format('census'),
                           table_size=48842))

    return schema


def gen_dmv_schema(csv_path):
    schema = SchemaGraph()

    schema.add_table(Table('dmv', attributes=['Record_Type', 'Registration_Class', 'State', 'County', 'Body_Type',
                                              'Fuel_Type', 'Reg_Valid_Date', 'Color', 'Scofflaw_Indicator',
                                              'Suspension_Indicator', 'Revocation_Indicator', ],
                           irrelevant_attributes=[],
                           no_compression=[],
                           csv_file_location=csv_path.format('dmv'),
                           table_size=11591877))

    return schema


def gen_cup98_10_schema(csv_path):
    schema = SchemaGraph()

    schema.add_table(Table('cup98_10', attributes=['AGE901', 'AGE902','AGE903', 'AGE904', 'AGE905',
                                                   'AGE906', 'AGE907', 'CHIL1', 'CHIL2', 'CHIL3', ],
                           irrelevant_attributes=[],
                           no_compression=[],
                           csv_file_location=csv_path.format('cup98_10'),
                           table_size=95412))

    return schema


def gen_cup98_20_schema(csv_path):
    schema = SchemaGraph()

    schema.add_table(Table('cup98_20', attributes=['AGE901', 'AGE902','AGE903', 'AGE904', 'AGE905',
                                                   'AGE906', 'AGE907', 'CHIL1', 'CHIL2', 'CHIL3',
                                                   'AGEC1',	'AGEC2', 'AGEC3', 'AGEC4', 'AGEC5',
                                                   'AGEC6','AGEC7',	'CHILC1', 'CHILC2', 'CHILC3',],
                           irrelevant_attributes=[],
                           no_compression=[],
                           csv_file_location=csv_path.format('cup98_20'),
                           table_size=95412))

    return schema


def gen_cup98_30_schema(csv_path):
    schema = SchemaGraph()

    schema.add_table(Table('cup98_30', attributes=['AGE901', 'AGE902','AGE903', 'AGE904', 'AGE905',
                                                   'AGE906', 'AGE907', 'CHIL1', 'CHIL2', 'CHIL3',
                                                   'AGEC1',	'AGEC2', 'AGEC3', 'AGEC4', 'AGEC5',
                                                   'AGEC6','AGEC7',	'CHILC1', 'CHILC2', 'CHILC3',
                                                   'CHILC4', 'CHILC5', 'HHAGE1', 'HHAGE2', 'HHAGE3',
                                                   'HHN1', 'HHN2', 'HHN3', 'HHN4', 'HHN5',],
                           irrelevant_attributes=[],
                           no_compression=[],
                           csv_file_location=csv_path.format('cup98_30'),
                           table_size=95412))

    return schema

def gen_kddcup98_schema(csv_path):
    schema = SchemaGraph()

    schema.add_table(Table('kddcup98', attributes=['ODATEDW', 'OSOURCE', 'TCODE', 'STATE', 'ZIP', 'MAILCODE', 'PVASTATE', 'DOB', 'NOEXCH', 'RECINHSE', 'RECP3', 'RECPGVG', 'RECSWEEP', 'MDMAUD', 'DOMAIN', 'CLUSTER', 'AGE', 'AGEFLAG', 'HOMEOWNR', 'CHILD03', 'CHILD07', 'CHILD12', 'CHILD18', 'NUMCHLD', 'INCOME', 'GENDER', 'WEALTH1', 'HIT', 'MBCRAFT', 'MBGARDEN', 'MBBOOKS', 'MBCOLECT', 'MAGFAML', 'MAGFEM', 'MAGMALE', 'PUBGARDN', 'PUBCULIN', 'PUBHLTH', 'PUBDOITY', 'PUBNEWFN', 'PUBPHOTO', 'PUBOPP', 'DATASRCE', 'MALEMILI', 'MALEVET', 'VIETVETS', 'WWIIVETS', 'LOCALGOV', 'STATEGOV', 'FEDGOV', 'SOLP3', 'SOLIH', 'MAJOR', 'WEALTH2', 'GEOCODE', 'COLLECT1', 'VETERANS', 'BIBLE', 'CATLG', 'HOMEE', 'PETS', 'CDPLAY', 'STEREO', 'PCOWNERS', 'PHOTO', 'CRAFTS', 'FISHER', 'GARDENIN', 'BOATS', 'WALKER', 'KIDSTUFF', 'CARDS', 'PLATES', 'LIFESRC', 'PEPSTRFL', 'POP901', 'POP902', 'POP903', 'POP90C1', 'POP90C2', 'POP90C3', 'POP90C4', 'POP90C5', 'ETH1', 'ETH2', 'ETH3', 'ETH4', 'ETH5', 'ETH6', 'ETH7', 'ETH8', 'ETH9', 'ETH10', 'ETH11', 'ETH12', 'ETH13', 'ETH14', 'ETH15', 'ETH16', 'AGE901', 'AGE902', 'AGE903', 'AGE904', 'AGE905', 'AGE906', 'AGE907', 'CHIL1', 'CHIL2', 'CHIL3', 'AGEC1', 'AGEC2', 'AGEC3', 'AGEC4', 'AGEC5', 'AGEC6', 'AGEC7', 'CHILC1', 'CHILC2', 'CHILC3', 'CHILC4', 'CHILC5', 'HHAGE1', 'HHAGE2', 'HHAGE3', 'HHN1', 'HHN2', 'HHN3', 'HHN4', 'HHN5', 'HHN6', 'MARR1', 'MARR2', 'MARR3', 'MARR4', 'HHP1', 'HHP2', 'DW1', 'DW2', 'DW3', 'DW4', 'DW5', 'DW6', 'DW7', 'DW8', 'DW9', 'HV1', 'HV2', 'HV3', 'HV4', 'HU1', 'HU2', 'HU3', 'HU4', 'HU5', 'HHD1', 'HHD2', 'HHD3', 'HHD4', 'HHD5', 'HHD6', 'HHD7', 'HHD8', 'HHD9', 'HHD10', 'HHD11', 'HHD12', 'ETHC1', 'ETHC2', 'ETHC3', 'ETHC4', 'ETHC5', 'ETHC6', 'HVP1', 'HVP2', 'HVP3', 'HVP4', 'HVP5', 'HVP6', 'HUR1', 'HUR2', 'RHP1', 'RHP2', 'RHP3', 'RHP4', 'HUPA1', 'HUPA2', 'HUPA3', 'HUPA4', 'HUPA5', 'HUPA6', 'HUPA7', 'RP1', 'RP2', 'RP3', 'RP4', 'MSA', 'ADI', 'DMA', 'IC1', 'IC2', 'IC3', 'IC4', 'IC5', 'IC6', 'IC7', 'IC8', 'IC9', 'IC10', 'IC11', 'IC12', 'IC13', 'IC14', 'IC15', 'IC16', 'IC17', 'IC18', 'IC19', 'IC20', 'IC21', 'IC22', 'IC23', 'HHAS1', 'HHAS2', 'HHAS3', 'HHAS4', 'MC1', 'MC2', 'MC3', 'TPE1', 'TPE2', 'TPE3', 'TPE4', 'TPE5', 'TPE6', 'TPE7', 'TPE8', 'TPE9', 'PEC1', 'PEC2', 'TPE10', 'TPE11', 'TPE12', 'TPE13', 'LFC1', 'LFC2', 'LFC3', 'LFC4', 'LFC5', 'LFC6', 'LFC7', 'LFC8', 'LFC9', 'LFC10', 'OCC1', 'OCC2', 'OCC3', 'OCC4', 'OCC5', 'OCC6', 'OCC7', 'OCC8', 'OCC9', 'OCC10', 'OCC11', 'OCC12', 'OCC13', 'EIC1', 'EIC2', 'EIC3', 'EIC4', 'EIC5', 'EIC6', 'EIC7', 'EIC8', 'EIC9', 'EIC10', 'EIC11', 'EIC12', 'EIC13', 'EIC14', 'EIC15', 'EIC16', 'OEDC1', 'OEDC2', 'OEDC3', 'OEDC4', 'OEDC5', 'OEDC6', 'OEDC7', 'EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6', 'EC7', 'EC8', 'SEC1', 'SEC2', 'SEC3', 'SEC4', 'SEC5', 'AFC1', 'AFC2', 'AFC3', 'AFC4', 'AFC5', 'AFC6', 'VC1', 'VC2', 'VC3', 'VC4', 'ANC1', 'ANC2', 'ANC3', 'ANC4', 'ANC5', 'ANC6', 'ANC7', 'ANC8', 'ANC9', 'ANC10', 'ANC11', 'ANC12', 'ANC13', 'ANC14', 'ANC15', 'POBC1', 'POBC2', 'LSC1', 'LSC2', 'LSC3', 'LSC4', 'VOC1', 'VOC2', 'VOC3', 'HC1', 'HC2', 'HC3', 'HC4', 'HC5', 'HC6', 'HC7', 'HC8', 'HC9', 'HC10', 'HC11', 'HC12', 'HC13', 'HC14', 'HC15', 'HC16', 'HC17', 'HC18', 'HC19', 'HC20', 'HC21', 'MHUC1', 'MHUC2', 'AC1', 'AC2', 'ADATE_2', 'ADATE_3', 'ADATE_4', 'ADATE_5', 'ADATE_6', 'ADATE_7', 'ADATE_8', 'ADATE_9', 'ADATE_10', 'ADATE_11', 'ADATE_12', 'ADATE_13', 'ADATE_14', 'ADATE_15', 'ADATE_16', 'ADATE_17', 'ADATE_18', 'ADATE_19', 'ADATE_20', 'ADATE_21', 'ADATE_22', 'ADATE_23', 'ADATE_24', 'RFA_2', 'RFA_3', 'RFA_4', 'RFA_5', 'RFA_6', 'RFA_7', 'RFA_8', 'RFA_9', 'RFA_10', 'RFA_11', 'RFA_12', 'RFA_13', 'RFA_14', 'RFA_15', 'RFA_16', 'RFA_17', 'RFA_18', 'RFA_19', 'RFA_20', 'RFA_21', 'RFA_22', 'RFA_23', 'RFA_24', 'CARDPROM', 'MAXADATE', 'NUMPROM', 'CARDPM12', 'NUMPRM12', 'RDATE_3', 'RDATE_4', 'RDATE_5', 'RDATE_6', 'RDATE_7', 'RDATE_8', 'RDATE_9', 'RDATE_10', 'RDATE_11', 'RDATE_12', 'RDATE_13', 'RDATE_14', 'RDATE_15', 'RDATE_16', 'RDATE_17', 'RDATE_18', 'RDATE_19', 'RDATE_20', 'RDATE_21', 'RDATE_22', 'RDATE_23', 'RDATE_24', 'RAMNT_3', 'RAMNT_4', 'RAMNT_5', 'RAMNT_6', 'RAMNT_7', 'RAMNT_8', 'RAMNT_9', 'RAMNT_10', 'RAMNT_11', 'RAMNT_12', 'RAMNT_13', 'RAMNT_14', 'RAMNT_15', 'RAMNT_16', 'RAMNT_17', 'RAMNT_18', 'RAMNT_19', 'RAMNT_20', 'RAMNT_21', 'RAMNT_22', 'RAMNT_23', 'RAMNT_24', 'RAMNTALL', 'NGIFTALL', 'CARDGIFT', 'MINRAMNT', 'MINRDATE', 'MAXRAMNT', 'MAXRDATE', 'LASTGIFT', 'LASTDATE', 'FISTDATE', 'NEXTDATE', 'TIMELAG', 'AVGGIFT', 'CONTROLN', 'TARGET_B', 'TARGET_D', 'HPHONE_D', 'RFA_2R', 'RFA_2F', 'RFA_2A', 'MDMAUD_R', 'MDMAUD_F', 'MDMAUD_A', 'CLUSTER2', 'GEOCODE2'],
                           irrelevant_attributes=['ODATEDW', 'OSOURCE', 'TCODE', 'ZIP', 'DOB', 'NOEXCH', 'AGE', 'NUMCHLD', 'INCOME', 'WEALTH1', 'HIT', 'MBCRAFT', 'MBGARDEN', 'MBBOOKS', 'MBCOLECT', 'MAGFAML', 'MAGFEM', 'MAGMALE', 'PUBGARDN', 'PUBCULIN', 'PUBHLTH', 'PUBDOITY', 'PUBNEWFN', 'PUBPHOTO', 'PUBOPP', 'MALEMILI', 'MALEVET', 'VIETVETS', 'WWIIVETS', 'LOCALGOV', 'STATEGOV', 'FEDGOV', 'WEALTH2', 'POP901', 'POP902', 'POP903', 'POP90C1', 'POP90C2', 'POP90C3', 'POP90C4', 'POP90C5', 'ETH1', 'ETH2', 'ETH3', 'ETH4', 'ETH5', 'ETH7', 'ETH8', 'ETH9', 'ETH13', 'ETH15', 'ETH16', 'AGE901', 'AGE902', 'AGE903', 'AGE904', 'AGE905', 'AGE906', 'AGE907', 'CHIL1', 'CHIL2', 'CHIL3', 'AGEC1', 'AGEC2', 'AGEC3', 'AGEC4', 'AGEC5', 'AGEC6', 'AGEC7', 'CHILC1', 'CHILC2', 'CHILC3', 'CHILC4', 'CHILC5', 'HHAGE1', 'HHAGE2', 'HHAGE3', 'HHN1', 'HHN2', 'HHN3', 'HHN4', 'HHN5', 'HHN6', 'MARR1', 'MARR2', 'MARR3', 'MARR4', 'HHP1', 'HHP2', 'DW1', 'DW2', 'DW3', 'DW4', 'DW5', 'DW6', 'DW7', 'DW8', 'DW9', 'HV1', 'HV2', 'HU1', 'HU2', 'HU3', 'HU4', 'HU5', 'HHD1', 'HHD2', 'HHD3', 'HHD4', 'HHD5', 'HHD6', 'HHD7', 'HHD9', 'HHD10', 'HHD11', 'HHD12', 'ETHC1', 'ETHC2', 'ETHC3', 'ETHC4', 'ETHC5', 'ETHC6', 'HVP1', 'HVP2', 'HVP3', 'HVP4', 'HVP5', 'HVP6', 'HUR1', 'HUR2', 'RHP1', 'RHP2', 'HUPA1', 'HUPA2', 'HUPA3', 'HUPA4', 'HUPA5', 'HUPA6', 'HUPA7', 'RP1', 'RP2', 'RP3', 'RP4', 'MSA', 'ADI', 'DMA', 'IC1', 'IC2', 'IC3', 'IC4', 'IC5', 'IC6', 'IC7', 'IC8', 'IC9', 'IC10', 'IC11', 'IC14', 'IC15', 'IC16', 'IC17', 'IC18', 'IC19', 'IC20', 'IC21', 'IC23', 'HHAS1', 'HHAS2', 'HHAS3', 'HHAS4', 'MC1', 'MC2', 'MC3', 'TPE1', 'TPE2', 'TPE3', 'TPE4', 'TPE5', 'TPE8', 'TPE9', 'PEC1', 'PEC2', 'TPE10', 'TPE11', 'TPE12', 'TPE13', 'LFC1', 'LFC2', 'LFC3', 'LFC4', 'LFC5', 'LFC6', 'LFC7', 'LFC8', 'LFC9', 'LFC10', 'OCC1', 'OCC2', 'OCC4', 'OCC5', 'OCC8', 'OCC9', 'OCC10', 'OCC11', 'OCC12', 'EIC1', 'EIC2', 'EIC3', 'EIC4', 'EIC5', 'EIC8', 'EIC9', 'EIC10', 'EIC11', 'EIC13', 'EIC14', 'EIC15', 'EIC16', 'OEDC1', 'OEDC2', 'OEDC3', 'OEDC4', 'OEDC5', 'OEDC6', 'EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC7', 'EC8', 'SEC1', 'SEC2', 'SEC4', 'SEC5', 'AFC1', 'AFC2', 'AFC4', 'AFC5', 'VC1', 'VC2', 'VC3', 'VC4', 'ANC1', 'ANC2', 'ANC4', 'ANC9', 'ANC10', 'POBC1', 'POBC2', 'LSC1', 'LSC2', 'LSC3', 'LSC4', 'VOC1', 'VOC2', 'VOC3', 'HC2', 'HC3', 'HC4', 'HC5', 'HC6', 'HC7', 'HC8', 'HC9', 'HC10', 'HC11', 'HC12', 'HC13', 'HC14', 'HC16', 'HC17', 'HC18', 'HC19', 'HC20', 'HC21', 'ADATE_3', 'ADATE_4', 'ADATE_5', 'ADATE_6', 'ADATE_7', 'ADATE_8', 'ADATE_9', 'ADATE_10', 'ADATE_11', 'ADATE_12', 'ADATE_13', 'ADATE_14', 'ADATE_15', 'ADATE_16', 'ADATE_17', 'ADATE_18', 'ADATE_19', 'ADATE_20', 'ADATE_21', 'ADATE_22', 'ADATE_23', 'ADATE_24', 'RFA_3', 'RFA_4', 'RFA_6', 'RFA_7', 'RFA_8', 'RFA_9', 'RFA_10', 'RFA_11', 'RFA_12', 'RFA_13', 'RFA_14', 'RFA_16', 'RFA_17', 'RFA_18', 'RFA_19', 'RFA_20', 'RFA_21', 'RFA_22', 'RFA_23', 'RFA_24', 'CARDPROM', 'NUMPROM', 'NUMPRM12', 'RDATE_3', 'RDATE_4', 'RDATE_5', 'RDATE_6', 'RDATE_7', 'RDATE_8', 'RDATE_9', 'RDATE_10', 'RDATE_11', 'RDATE_12', 'RDATE_13', 'RDATE_14', 'RDATE_15', 'RDATE_16', 'RDATE_17', 'RDATE_18', 'RDATE_19', 'RDATE_20', 'RDATE_21', 'RDATE_22', 'RDATE_23', 'RDATE_24', 'RAMNT_3', 'RAMNT_4', 'RAMNT_5', 'RAMNT_6', 'RAMNT_7', 'RAMNT_8', 'RAMNT_9', 'RAMNT_10', 'RAMNT_11', 'RAMNT_12', 'RAMNT_13', 'RAMNT_14', 'RAMNT_15', 'RAMNT_16', 'RAMNT_17', 'RAMNT_18', 'RAMNT_19', 'RAMNT_20', 'RAMNT_21', 'RAMNT_22', 'RAMNT_23', 'RAMNT_24', 'RAMNTALL', 'NGIFTALL', 'CARDGIFT', 'MINRAMNT', 'MINRDATE', 'MAXRAMNT', 'MAXRDATE', 'LASTGIFT', 'FISTDATE', 'NEXTDATE', 'TIMELAG', 'AVGGIFT', 'CONTROLN', 'TARGET_D', 'CLUSTER2', 'GEOCODE2'],
                           no_compression=[],
                           csv_file_location=csv_path.format('kddcup98'),
                           table_size=95412))

    return schema


def gen_ssb_schema(csv_path):
    """
    Just like the full IMDB schema but without tables that are not used in the job-light benchmark.
    """

    schema = SchemaGraph()

    # tables

    # customer
    schema.add_table(Table('customer', attributes=['c_custkey', 'c_name', 'c_address', 'c_city', 'c_nation', 'c_region',
                                                   'c_phone', 'c_mktsegment', 'dummy'],
                           irrelevant_attributes=['c_name', 'c_address',
                                                   'c_phone', 'c_mktsegment', 'dummy'],
                           no_compression=[],
                           primary_key=['c_custkey'],
                           csv_file_location=csv_path.format('customer'),
                           table_size=300000))
    # table_size=2524974))
    ### table_size=3486660)

    # supplier
    schema.add_table(Table('supplier', attributes=['s_suppkey', 's_name', 's_address', 's_city', 's_nation', 's_region',
                                                   's_phone', 'dummy'],
                           csv_file_location=csv_path.format('supplier'),
                           irrelevant_attributes=['s_name', 's_address',
                                                  's_phone', 'dummy'],
                           no_compression=[],
                           primary_key=['s_suppkey'],
                           table_size=20000))
    # table_size=1380035))
    ### table_size=3147110

    # lineorder
    schema.add_table(Table('lineorder', attributes=['lo_orderkey', 'lo_linenumber', 'lo_custkey', 'lo_partkey', 'lo_suppkey',
                                                    'lo_orderdate', 'lo_orderpriority', 'lo_shippriority', 'lo_quantity',
                                                    'lo_extendedprice', 'lo_ordtotalprice', 'lo_discount', 'lo_revenue',
                                                    'lo_supplycost', 'lo_tax', 'lo_commitdate', 'lo_shipmod', 'dummy'],
                           csv_file_location=csv_path.format('lineorder'),
                           irrelevant_attributes=['lo_orderpriority', 'lo_shippriority',
                                                  'lo_extendedprice', 'lo_ordtotalprice', 'lo_revenue',
                                                  'lo_supplycost', 'lo_tax', 'lo_commitdate', 'lo_shipmod', 'dummy'],
                           no_compression=[],
                           # primary_key=['lo_orderkey', 'lo_linenumber'],
                           primary_key=['lo_orderkey'],
                           table_size=59986214))
    # table_size=14615575))
    ### table_size=24988000)

    # part
    schema.add_table(Table('part', attributes=['p_partkey', 'p_name', 'p_mfgr', 'p_category', 'p_brand1', 'p_color',
                                               'p_type', 'p_size', 'p_container', 'dummy'],
                           csv_file_location=csv_path.format('part'),
                           irrelevant_attributes=['p_name', 'p_color',
                                                  'p_type', 'p_size', 'p_container', 'dummy'],
                           no_compression=[],
                           primary_key=['p_partkey'],
                           table_size=800000))
    # table_size=36243321
    ### table_size=63475800

    # date
    schema.add_table(Table('dwdate', attributes=['d_datekey', 'd_date', 'd_dayofweek', 'd_month', 'd_year', 'd_yearmonthnum',
                                               'd_yearmonth', 'd_daynuminweek', 'd_daynuminmonth', 'd_daynuminyear',
                                               'd_monthnuminyear', 'd_weeknuminyear', 'd_sellingseason', 'd_lastdayinweekfl',
                                               'd_lastdayinmonthfl', 'd_holidayfl', 'd_weekdayfl', 'dummy'],
                           csv_file_location=csv_path.format('dwdate'),
                           irrelevant_attributes=['d_date', 'd_dayofweek', 'd_month','d_daynuminweek',
                                                  'd_daynuminmonth', 'd_daynuminyear',
                                                  'd_monthnuminyear', 'd_sellingseason', 'd_lastdayinweekfl',
                                                  'd_lastdayinmonthfl', 'd_holidayfl', 'd_weekdayfl', 'dummy'],
                           no_compression=[],
                           primary_key=['d_datekey'],
                           table_size=2556))


    # relationships
    schema.add_relationship('lineorder', 'lo_orderdate', 'dwdate', 'd_datekey')
    schema.add_relationship('lineorder', 'lo_partkey', 'part', 'p_partkey')
    schema.add_relationship('lineorder', 'lo_custkey', 'customer', 'c_custkey')
    schema.add_relationship('lineorder', 'lo_suppkey', 'supplier', 's_suppkey')
    # schema.add_relationship('movie_companies', 'movie_id', 'title', 'id')

    ###
    # schema.add_relationship('dwdate', 'd_datekey', 'lineorder', 'lo_orderdate')
    # schema.add_relationship('part', 'p_partkey', 'lineorder', 'lo_partkey')
    # schema.add_relationship('customer', 'c_custkey', 'lineorder', 'lo_custkey')
    # schema.add_relationship('supplier', 's_suppkey', 'lineorder', 'lo_suppkey')

    return schema


def gen_job_light_imdb_schema(csv_path):
    """
    Just like the full IMDB schema but without tables that are not used in the job-light benchmark.
    """

    schema = SchemaGraph()

    # tables

    # title
    schema.add_table(Table('title', attributes=['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id',
                                                'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr',
                                                'series_years', 'md5sum'],
                           irrelevant_attributes=['episode_of_id', 'title', 'imdb_index', 'phonetic_code', 'season_nr',
                                                  'imdb_id', 'episode_nr', 'series_years', 'md5sum'],
                           no_compression=['kind_id'],
                           csv_file_location=csv_path.format('title'),
                           table_size=2097822))
                            # table_size=2524974))
                            ### table_size=3486660)

    # movie_info_idx
    schema.add_table(Table('movie_info_idx', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('movie_info_idx'),
                           irrelevant_attributes=['info', 'note'],
                           no_compression=['info_type_id'],
                           table_size=1380035))
    # table_size=1380035))
    ### table_size=3147110

    # movie_info
    schema.add_table(Table('movie_info', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('movie_info'),
                           irrelevant_attributes=['info', 'note'],
                           no_compression=['info_type_id'],
                           table_size=11880517))
    # table_size=14615575))
    ### table_size=24988000)

    # cast_info
    schema.add_table(Table('cast_info', attributes=['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order',
                                                    'role_id'],
                           csv_file_location=csv_path.format('cast_info'),
                           irrelevant_attributes=['nr_order', 'note', 'person_id', 'person_role_id'],
                           no_compression=['role_id'],
                           table_size=35691620))
    # table_size=36243321
    ### table_size=63475800

    # movie_keyword
    schema.add_table(Table('movie_keyword', attributes=['id', 'movie_id', 'keyword_id'],
                           csv_file_location=csv_path.format('movie_keyword'),
                           no_compression=['keyword_id'],
                           table_size=4523930))
    # table_size=4523930))
    ### table_size=7522600

    # movie_companies
    schema.add_table(Table('movie_companies', attributes=['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                           csv_file_location=csv_path.format('movie_companies'),
                           irrelevant_attributes=['note'],
                           no_compression=['company_id', 'company_type_id'],
                           table_size=2609129))
    ### table_size=4958300

    # relationships
    schema.add_relationship('movie_info_idx', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_info', 'movie_id', 'title', 'id')
    schema.add_relationship('cast_info', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_keyword', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_companies', 'movie_id', 'title', 'id')

    return schema


def gen_stats_ceb_schema(csv_path):
    """
    Just like the full IMDB schema but without tables that are not used in the job-light benchmark.
    """

    schema = SchemaGraph()

    # tables

    # posts
    schema.add_table(Table('posts', attributes=['Id', 'PostTypeId', 'CreationDate', 'Score', 'ViewCount', 'OwnerUserId',
                                                'AnswerCount', 'CommentCount', 'FavoriteCount', 'LastEditorUserId'],
                           irrelevant_attributes=[],
                           no_compression=['Id', 'PostTypeId', 'OwnerUserId', 'LastEditorUserId'],
                           csv_file_location=csv_path.format('posts'),
                           table_size=91976,
                           primary_key=["Id"]))
    ### table_size=3486660)

    # users
    schema.add_table(Table('users', attributes=['Id', 'Reputation', 'CreationDate', 'Views', 'UpVotes', 'DownVotes'],
                           csv_file_location=csv_path.format('users'),
                           irrelevant_attributes=[],
                           no_compression=['Id'],
                           table_size=40325,
                           primary_key=["Id"]))
    ### table_size=3147110

    # tags
    schema.add_table(Table('tags', attributes=['Id', 'Count', 'ExcerptPostId'],
                           csv_file_location=csv_path.format('tags'),
                           irrelevant_attributes=[],
                           no_compression=['Id', 'ExcerptPostId'],
                           table_size=1032,
                           primary_key=["Id"]))
    ### table_size=24988000)

    # comments
    schema.add_table(Table('comments', attributes=['Id', 'PostId', 'Score', 'CreationDate', 'UserId'],
                           csv_file_location=csv_path.format('comments'),
                           irrelevant_attributes=[],
                           no_compression=['Id', 'PostId', 'UserId'],
                           table_size=174305,
                           primary_key=["Id"]))
    ### table_size=63475800

    # badges
    schema.add_table(Table('badges', attributes=['Id', 'UserId', 'Date'],
                           csv_file_location=csv_path.format('badges'),
                           no_compression=['Id', 'UserId'],
                           table_size=79851,
                           primary_key=["Id"]))
    ### table_size=7522600

    # postLinks
    schema.add_table(Table('postLinks', attributes=['Id', 'CreationDate', 'PostId', 'RelatedPostId', 'LinkTypeId'],
                           csv_file_location=csv_path.format('postLinks'),
                           irrelevant_attributes=[],
                           no_compression=['Id', 'PostId', 'RelatedPostId', 'LinkTypeId'],
                           table_size=11102,
                           primary_key=["Id"]))
    ### table_size=4958300

    # postHistory
    schema.add_table(Table('postHistory', attributes=['Id', 'PostHistoryTypeId', 'PostId', 'CreationDate', 'UserId'],
                           csv_file_location=csv_path.format('postHistory'),
                           irrelevant_attributes=[],
                           no_compression=['Id', 'PostHistoryTypeId', 'PostId', 'UserId'],
                           table_size=303187,
                           primary_key=["Id"]))

    # votes
    schema.add_table(Table('votes', attributes=['Id', 'PostId', 'VoteTypeId', 'CreationDate', 'UserId', 'BountyAmount'],
                           csv_file_location=csv_path.format('votes'),
                           irrelevant_attributes=[],
                           no_compression=['Id', 'PostId', 'VoteTypeId', 'UserId'],
                           table_size=328604,
                           primary_key=["Id"]))

    # relationships
    schema.add_relationship('tags', 'ExcerptPostId', 'posts', 'Id')
    schema.add_relationship('comments', 'PostId', 'posts', 'Id')
    schema.add_relationship('comments', 'UserId', 'users', 'Id')
    schema.add_relationship('badges', 'UserId', 'users', 'Id')
    schema.add_relationship('posts', 'OwnerUserId', 'users', 'Id')
    schema.add_relationship('postLinks', 'PostId', 'posts', 'Id')
    schema.add_relationship('postHistory', 'PostId', 'posts', 'Id')
    schema.add_relationship('postHistory', 'UserId', 'users', 'Id')
    schema.add_relationship('votes', 'UserId', 'users', 'Id')
    schema.add_relationship('votes', 'PostId', 'posts', 'Id')

    ###
    schema.add_relationship('postLinks', 'RelatedPostId', 'posts', 'Id')
    schema.add_relationship('posts', 'LastEditorUserId', 'users', 'Id')

    return schema


def gen_imdb_schema(csv_path):
    """
    Specifies full imdb-benchmark schema. Also tables not in the job-light benchmark.
    """
    schema = SchemaGraph()

    # tables

    # title
    schema.add_table(Table('title', attributes=['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id',
                                                'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr',
                                                'series_years', 'md5sum'],
                           irrelevant_attributes=['episode_of_id'],
                           csv_file_location=csv_path.format('title'),
                           table_size=3486660))

    # movie_info_idx
    schema.add_table(Table('movie_info_idx', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('movie_info_idx'),
                           table_size=3147110))

    # movie_info
    schema.add_table(Table('movie_info', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('movie_info'),
                           table_size=24988000))

    # info_type
    schema.add_table(Table('info_type', attributes=['id', 'info'],
                           csv_file_location=csv_path.format('info_type'),
                           table_size=113))

    # cast_info
    schema.add_table(Table('cast_info', attributes=['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order',
                                                    'role_id'],
                           csv_file_location=csv_path.format('cast_info'),
                           table_size=63475800))

    # char_name
    schema.add_table(Table('char_name', attributes=['id', 'name', 'imdb_index', 'imdb_id', 'name_pcode_nf',
                                                    'surname_pcode', 'md5sum'],
                           csv_file_location=csv_path.format('char_name'),
                           table_size=4314870))

    # role_type
    schema.add_table(Table('role_type', attributes=['id', 'role'],
                           csv_file_location=csv_path.format('role_type'),
                           table_size=0))

    # complete_cast
    schema.add_table(Table('complete_cast', attributes=['id', 'movie_id', 'subject_id', 'status_id'],
                           csv_file_location=csv_path.format('complete_cast'),
                           table_size=135086))

    # comp_cast_type
    schema.add_table(Table('comp_cast_type', attributes=['id', 'kind'],
                           csv_file_location=csv_path.format('comp_cast_type'),
                           table_size=0))

    # name
    schema.add_table(Table('name', attributes=['id', 'name', 'imdb_index', 'imdb_id', 'gender', 'name_pcode_cf',
                                               'name_pcode_nf', 'surname_pcode', 'md5sum'],
                           csv_file_location=csv_path.format('name'),
                           table_size=6379740))

    # aka_name
    schema.add_table(Table('aka_name', attributes=['id', 'person_id', 'name', 'imdb_index', 'name_pcode_cf',
                                                   'name_pcode_nf', 'surname_pcode', 'md5sum'],
                           csv_file_location=csv_path.format('aka_name'),
                           table_size=1312270))

    # movie_link, is empty
    # schema.add_table(Table('movie_link', attributes=['id', 'movie_id', 'linked_movie_id', 'link_type_id'],
    #                        csv_file_location=csv_path.format('movie_link')))

    # link_type, no relationships
    # schema.add_table(Table('link_type', attributes=['id', 'link'],
    #                        csv_file_location=csv_path.format('link_type')))

    # movie_keyword
    schema.add_table(Table('movie_keyword', attributes=['id', 'movie_id', 'keyword_id'],
                           csv_file_location=csv_path.format('movie_keyword'),
                           table_size=7522600))

    # keyword
    schema.add_table(Table('keyword', attributes=['id', 'keyword', 'phonetic_code'],
                           csv_file_location=csv_path.format('keyword'),
                           table_size=236627))

    # person_info
    schema.add_table(Table('person_info', attributes=['id', 'person_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('person_info'),
                           table_size=4130210))

    # movie_companies
    schema.add_table(Table('movie_companies', attributes=['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                           csv_file_location=csv_path.format('movie_companies'),
                           table_size=4958300))

    # company_name
    schema.add_table(Table('company_name', attributes=['id', 'name', 'country_code', 'imdb_id', 'name_pcode_nf',
                                                       'name_pcode_sf', 'md5sum'],
                           csv_file_location=csv_path.format('company_name'),
                           table_size=362131))

    # company_type
    schema.add_table(Table('company_type', attributes=['id', 'kind'],
                           csv_file_location=csv_path.format('company_type'),
                           table_size=0))

    # aka_title
    schema.add_table(Table('aka_title', attributes=['id', 'movie_id', 'title', 'imdb_index', 'kind_id',
                                                    'production_year', 'phonetic_code', 'episode_of_id', 'season_nr',
                                                    'episode_nr', 'note', 'md5sum'],
                           irrelevant_attributes=['episode_of_id'],
                           csv_file_location=csv_path.format('aka_title'),
                           table_size=528268))

    # kind_type
    schema.add_table(Table('kind_type', attributes=['id', 'kind'],
                           csv_file_location=csv_path.format('kind_type'),
                           table_size=7))

    # relationships

    # title
    # omit self-join for now
    # schema.add_relationship('title', 'episode_of_id', 'title', 'id')
    schema.add_relationship('title', 'kind_id', 'kind_type', 'id')

    # movie_info_idx
    schema.add_relationship('movie_info_idx', 'info_type_id', 'info_type', 'id')
    schema.add_relationship('movie_info_idx', 'movie_id', 'title', 'id')

    # movie_info
    schema.add_relationship('movie_info', 'info_type_id', 'info_type', 'id')
    schema.add_relationship('movie_info', 'movie_id', 'title', 'id')

    # info_type, no relationships

    # cast_info
    schema.add_relationship('cast_info', 'movie_id', 'title', 'id')
    schema.add_relationship('cast_info', 'person_id', 'name', 'id')
    schema.add_relationship('cast_info', 'person_role_id', 'char_name', 'id')
    schema.add_relationship('cast_info', 'role_id', 'role_type', 'id')

    # char_name, no relationships

    # role_type, no relationships

    # complete_cast
    schema.add_relationship('complete_cast', 'movie_id', 'title', 'id')
    schema.add_relationship('complete_cast', 'status_id', 'comp_cast_type', 'id')
    schema.add_relationship('complete_cast', 'subject_id', 'comp_cast_type', 'id')

    # comp_cast_type, no relationships

    # name, no relationships

    # aka_name
    schema.add_relationship('aka_name', 'person_id', 'name', 'id')

    # movie_link, is empty
    # schema.add_relationship('movie_link', 'link_type_id', 'link_type', 'id')
    # schema.add_relationship('movie_link', 'linked_movie_id', 'title', 'id')
    # schema.add_relationship('movie_link', 'movie_id', 'title', 'id')

    # link_type, no relationships

    # movie_keyword
    schema.add_relationship('movie_keyword', 'keyword_id', 'keyword', 'id')
    schema.add_relationship('movie_keyword', 'movie_id', 'title', 'id')

    # keyword, no relationships

    # person_info
    schema.add_relationship('person_info', 'info_type_id', 'info_type', 'id')
    schema.add_relationship('person_info', 'person_id', 'name', 'id')

    # movie_companies
    schema.add_relationship('movie_companies', 'company_id', 'company_name', 'id')
    schema.add_relationship('movie_companies', 'company_type_id', 'company_type', 'id')
    schema.add_relationship('movie_companies', 'movie_id', 'title', 'id')

    # company_name, no relationships

    # company_type, no relationships

    # aka_title
    schema.add_relationship('aka_title', 'movie_id', 'title', 'id')
    schema.add_relationship('aka_title', 'kind_id', 'kind_type', 'id')

    # kind_type, no relationships

    return schema


###
def gen_job_schema(csv_path):
    """
    Specifies full imdb-benchmark schema. Also tables not in the job-light benchmark.
    """
    schema = SchemaGraph()

    # tables

    # title
    schema.add_table(Table('title', attributes=['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id',
                                                'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr',
                                                'series_years', 'md5sum'],
                           # irrelevant_attributes=['episode_of_id'],
                           irrelevant_attributes=['imdb_index', 'imdb_id', 'phonetic_code', 'episode_of_id',
                                                  'season_nr', 'series_years', 'md5sum'],
                           csv_file_location=csv_path.format('title'),
                           table_size=3486660))

    # movie_info_idx
    schema.add_table(Table('movie_info_idx', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           irrelevant_attributes=['note'],
                           csv_file_location=csv_path.format('movie_info_idx'),
                           table_size=3147110))

    # movie_info
    schema.add_table(Table('movie_info', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           irrelevant_attributes=['info_type_id'],
                           csv_file_location=csv_path.format('movie_info'),
                           table_size=24988000))

    # info_type
    schema.add_table(Table('info_type', attributes=['id', 'info'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('info_type'),
                           table_size=113))

    # cast_info
    schema.add_table(Table('cast_info', attributes=['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order',
                                                    'role_id'],
                           irrelevant_attributes=['person_id', 'person_role_id', 'nr_order',
                                                  'role_id'],
                           csv_file_location=csv_path.format('cast_info'),
                           table_size=63475800))

    # char_name
    # schema.add_table(Table('char_name', attributes=['id', 'name', 'imdb_index', 'imdb_id', 'name_pcode_nf',
    #                                                 'surname_pcode', 'md5sum'],
    #                        csv_file_location=csv_path.format('char_name'),
    #                        table_size=4314870))

    # role_type
    # schema.add_table(Table('role_type', attributes=['id', 'role'],
    #                        csv_file_location=csv_path.format('role_type'),
    #                        table_size=0))

    # complete_cast
    schema.add_table(Table('complete_cast', attributes=['id', 'movie_id', 'subject_id', 'status_id'],
                           irrelevant_attributes=['status_id'],
                           csv_file_location=csv_path.format('complete_cast'),
                           table_size=135086))

    # comp_cast_type
    schema.add_table(Table('comp_cast_type', attributes=['id', 'kind'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('comp_cast_type'),
                           table_size=4))

    # # name
    # schema.add_table(Table('name', attributes=['id', 'name', 'imdb_index', 'imdb_id', 'gender', 'name_pcode_cf',
    #                                            'name_pcode_nf', 'surname_pcode', 'md5sum'],
    #                        csv_file_location=csv_path.format('name'),
    #                        table_size=6379740))

    # # aka_name
    # schema.add_table(Table('aka_name', attributes=['id', 'person_id', 'name', 'imdb_index', 'name_pcode_cf',
    #                                                'name_pcode_nf', 'surname_pcode', 'md5sum'],
    #                        csv_file_location=csv_path.format('aka_name'),
    #                        table_size=1312270))

    # movie_link, is empty
    schema.add_table(Table('movie_link', attributes=['id', 'movie_id', 'linked_movie_id', 'link_type_id'],
                           irrelevant_attributes=['linked_movie_id'],
                           csv_file_location=csv_path.format('movie_link'),
                           table_size=29997))

    # link_type, no relationships
    schema.add_table(Table('link_type', attributes=['id', 'link'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('link_type'),
                           table_size=18))

    # movie_keyword
    schema.add_table(Table('movie_keyword', attributes=['id', 'movie_id', 'keyword_id'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('movie_keyword'),
                           table_size=7522600))

    # keyword
    schema.add_table(Table('keyword', attributes=['id', 'keyword', 'phonetic_code'],
                           irrelevant_attributes=['phonetic_code'],
                           csv_file_location=csv_path.format('keyword'),
                           table_size=236627))

    # # person_info
    # schema.add_table(Table('person_info', attributes=['id', 'person_id', 'info_type_id', 'info', 'note'],
    #                        csv_file_location=csv_path.format('person_info'),
    #                        table_size=4130210))

    # movie_companies
    schema.add_table(Table('movie_companies', attributes=['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('movie_companies'),
                           table_size=4958300))

    # company_name
    schema.add_table(Table('company_name', attributes=['id', 'name', 'country_code', 'imdb_id', 'name_pcode_nf',
                                                       'name_pcode_sf', 'md5sum'],
                           irrelevant_attributes=['imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'],
                           csv_file_location=csv_path.format('company_name'),
                           table_size=362131))

    # company_type
    schema.add_table(Table('company_type', attributes=['id', 'kind'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('company_type'),
                           table_size=4))

    # aka_title
    schema.add_table(Table('aka_title', attributes=['id', 'movie_id', 'title', 'imdb_index', 'kind_id',
                                                    'production_year', 'phonetic_code', 'episode_of_id', 'season_nr',
                                                    'episode_nr', 'note', 'md5sum'],
                           irrelevant_attributes=['title', 'imdb_index', 'kind_id',
                                                  'production_year', 'phonetic_code', 'episode_of_id', 'season_nr',
                                                  'episode_nr', 'note', 'md5sum'],
                           csv_file_location=csv_path.format('aka_title'),
                           table_size=528268))

    # kind_type
    schema.add_table(Table('kind_type', attributes=['id', 'kind'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('kind_type'),
                           table_size=7))

    # relationships

    # title
    # omit self-join for now
    # schema.add_relationship('title', 'episode_of_id', 'title', 'id')
    schema.add_relationship('title', 'kind_id', 'kind_type', 'id')

    # movie_info_idx
    schema.add_relationship('movie_info_idx', 'info_type_id', 'info_type', 'id')
    schema.add_relationship('movie_info_idx', 'movie_id', 'title', 'id')

    # movie_info
    # ### schema.add_relationship('movie_info', 'info_type_id', 'info_type', 'id')
    schema.add_relationship('movie_info', 'movie_id', 'title', 'id')

    # info_type, no relationships

    # cast_info
    schema.add_relationship('cast_info', 'movie_id', 'title', 'id')
    # ### schema.add_relationship('cast_info', 'person_id', 'name', 'id')
    # ### schema.add_relationship('cast_info', 'person_role_id', 'char_name', 'id')
    # ### schema.add_relationship('cast_info', 'role_id', 'role_type', 'id')

    # char_name, no relationships

    # role_type, no relationships

    # complete_cast
    schema.add_relationship('complete_cast', 'movie_id', 'title', 'id')
    # ### schema.add_relationship('complete_cast', 'status_id', 'comp_cast_type', 'id')
    schema.add_relationship('complete_cast', 'subject_id', 'comp_cast_type', 'id')

    # comp_cast_type, no relationships

    # name, no relationships

    # aka_name
    # ### schema.add_relationship('aka_name', 'person_id', 'name', 'id')

    # movie_link, is empty
    schema.add_relationship('movie_link', 'link_type_id', 'link_type', 'id')
    # schema.add_relationship('movie_link', 'linked_movie_id', 'title', 'id')
    schema.add_relationship('movie_link', 'movie_id', 'title', 'id')

    # link_type, no relationships

    # movie_keyword
    schema.add_relationship('movie_keyword', 'keyword_id', 'keyword', 'id')
    schema.add_relationship('movie_keyword', 'movie_id', 'title', 'id')

    # keyword, no relationships

    # person_info
    # ### schema.add_relationship('person_info', 'info_type_id', 'info_type', 'id')
    # ### schema.add_relationship('person_info', 'person_id', 'name', 'id')

    # movie_companies
    schema.add_relationship('movie_companies', 'company_id', 'company_name', 'id')
    schema.add_relationship('movie_companies', 'company_type_id', 'company_type', 'id')
    schema.add_relationship('movie_companies', 'movie_id', 'title', 'id')

    # company_name, no relationships

    # company_type, no relationships

    # aka_title
    schema.add_relationship('aka_title', 'movie_id', 'title', 'id')
    # ### schema.add_relationship('aka_title', 'kind_id', 'kind_type', 'id')

    # kind_type, no relationships

    return schema
