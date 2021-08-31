#!/usr/bin/python
# -*- coding: utf-8 -*-

import itertools
import json
import logging
import math
import os,string,copy
import time
import re

import fasttext
from fasttext.FastText import supervised
import numpy as np
from numpy.core.numerictypes import ScalarType
import pandas as pd
from polyglot.detect import Detector
import requests
import json
import s3fs
import snowflake.connector
import ast
from collections import Counter  # available in Python 2.7 and newer
import collections


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

class LanguageDetectorTranslator:

# constructor
    def __init__(self):
        USER = os.environ.get("SNOWFLAKE_USER")
        PASSWORD = os.environ.get("SNOWFLAKE_PASSWORD")
        ACCOUNT = os.environ.get("SNOWFLAKE_ACCOUNT") 
        ROLE = os.environ.get("SNOWFLAKE_ROLE")
        self.country_codes = pd.read_csv(os.environ.get("COUNTRY_MAPPING_PATH"))
        self.translation_calls = 0
        self.failed_translation_calls = 0
        self.passby_translation_calls = 0
        self.CUSTOMER = os.environ.get("CUSTOMER") 
        self.DEPLOY_ENVIRONMENT = os.environ.get("DEPLOY_ENVIRONMENT") 
        self.INTENT_TABLE = os.environ.get("SNOWFLAKE_INTENT_TABLE") 
        self.LOOKUP_TABLE = os.environ.get("SNOWFLAKE_LOOKUP_TABLE")
        self.SCHEMA = os.environ.get("SCHEMA") 
        self.BKP_DEPLOY = os.environ.get("BKP_DEPLOY") 
        self.API_KEY = os.environ.get("API_KEY") 
        self.QUERY_LIMIT = os.environ.get("QUERY_LIMIT")
        self.SLACK_WEBHOOK = os.environ.get("SLACK_WEBHOOK")
        self.API_ENDPOINT = "https://translation.googleapis.com/language/translate/v2?target=en&key={1}&q={0}"
        self.NEW_KEYWORDS = os.environ.get("BASE_QUERY").format(
            self.DEPLOY_ENVIRONMENT, self.SCHEMA, self.INTENT_TABLE, self.QUERY_LIMIT
        )
        self.S3_BACKUP = "{0}_job_{1}{2}".format(
            os.environ.get("S3_BACKUP"), time.time(), ".csv.gzip"
        )
        self.BATCH_SIZE = int(os.environ.get("BATCH_SIZE") )
        self.s3 = s3fs.S3FileSystem(anon=False)
        # self.model = fasttext.load_model("./lid.176.ftz")
        self.temp_table = None

        try:
            self.con = snowflake.connector.connect(
                user=USER,
                password=PASSWORD,
                account=ACCOUNT,
                role=ROLE,
                session_parameters={"QUERY_TAG": "Loading data to intent"},
            )
            logging.info("Snowflake connection established successfully")
            # logging.info("BACK FILE REFERENCE %s" % self.S3_BACKUP)
        except snowflake.connector.errors.ProgrammingError as e:
            logging.error(
                "Got error while creating snowflake connection error = {}".format(e)
            )
        except Exception as e:
            logging.error(
                "Got exception while connnecting to snowflake error = {}".format(e)
            )

    def get_abbr_list(self):
        countries_list = ['afghanistan','albania','algeria','andorra','angola','antigua and barbuda','argentina','armenia'
                  ,'australia','austria','azerbaijan','the bahamas','bahrain','bangladesh','barbados','belarus'
                  ,'belgium','belize','benin','bhutan','bolivia','bosnia and herzegovina','botswana','brazil'
                  ,'brunei','bulgaria','burkina Faso','burundi','cabo Verde','cambodia','cameroon','canada'
                  ,'central African Republic','chad','chile','china','colombia','comoros','congo','costa Rica'
                  ,'côte d’Ivoire','croatia','cuba','cyprus','czech Republic','denmark','djibouti','dominica'
                  ,'dominican Republic','east Timor','ecuador','egypt','el salvador','equatorial guinea','eritrea'
                  ,'estonia','eswatini','ethiopia','fiji','finland','france','gabon','the gambia','georgia'
                  ,'germany','ghana','greece','grenada','guatemala','guinea','guinea-bissau','guyana','haiti'
                  ,'honduras','hungary','iceland','india','indonesia','iran','iraq','ireland','israel','italy'
                  ,'jamaica','japan','jordan','kazakhstan','kenya','kiribati','korea','korea','kosovo','kuwait'
                  ,'kyrgyzstan','laos','latvia','lebanon','lesotho','liberia','libya','liechtenstein','lithuania'
                  ,'luxembourg','madagascar','malawi','malaysia','maldives','mali','malta','marshall islands'
                  ,'mauritania','mauritius','mexico','micronesia','moldova','monaco','mongolia','montenegro'
                  ,'morocco','mozambique','myanmar' ,'namibia','nauru','nepal','netherlands','new zealand'
                  ,'nicaragua','niger','nigeria','north macedonia','norway','oman','pakistan','palau','panama'
                  ,'papua new guinea','paraguay','peru','philippines','poland','portugal','qatar','romania'
                  ,'russia','rwanda','saint kitts and nevis','saint lucia','saint vincent and the grenadines'
                  ,'samoa','san marino','sao tome and principe','saudi arabia','senegal','serbia','seychelles'
                  ,'sierra leone','singapore','slovakia','slovenia','solomon islands','somalia','south africa'
                  ,'spain','sri lanka','sudan','sudan','suriname','sweden','switzerland','syria','taiwan'
                  ,'tajikistan','tanzania','thailand','togo','tonga','trinidad and tobago','tunisia','turkey'
                  ,'turkmenistan','tuvalu','uganda','ukraine','united arab emirates','united kingdom'
                  ,'united states','uruguay','uzbekistan','vanuatu','vatican city','venezuela','vietnam'
                  ,'yemen','zambia','zimbabwe','colombia','italia','deutshland','deutsch','africa'
                  ,'francais','danmark','santa clara','amsterdam','hong kong','banglore','norge'
                  ,'mumbai','italiano','brasil','brazil','espanol','spanish','london','europ','ontario','dubai','kanda','deutschland']

        countries_abbr = ['hk','us','usa','uae','nz','uk','au','ca','de','sg','arg','fr','nl','mex'
                  ,'aust', 'it', 'finnish', 'se', 'be', 'mx', 'tw', 'ph', 'br', 'fi', 'ge'
                  ,'jp', 'pf', 'ru','bh','ch']
        abbr_list = ['il','novartis','novatis','novitis','ppi','tb','psa','fda','pso','pi','smn','ibd','ndc','ai','pi','cpt','spc','med','dx','din','ad','pap','icd','pt','srf','ra','ema','icd10','ny','uspi','ca','rx','s/c','pis','ind','nsclc','cyp','dvt','crp','moa','SAEs','cyp3a4','nct','asco','uspi','hcp','rcp','smpc','ema','chmp','aki','pi3k','cdk','braf','ggt','itp','png','her2','her2+','hr+','nct03828539','rep','op','nass','rems','arb','dro','cindy','lauper','canada','pis','H1047R','mek','BRAFV600E','brafv600e','dvt','nsclc','per','v600k','meke','v600','norvatis','ctcae','cyp3a4','braf-v600e','rcp','v600','testi','v600e','chemocare','egfr','braf600','braf-mek','v600f','faqs','faq','melanom','braftovi','erk','mapk','plm','basmi','lyo','a375','tbc','bid','vial','tbl','cyndi','pil','at&t','esm','wap','aws','htr-6072','snqpchat','pc','php','if','sdl','iud','iud,','ak','sd','wan','ist','tut','ccna','sase','pst','jst','ips','vpn','lan','sra','mpls','vs','iwan','sd-wan','sdwan','sdn','vnet','cmo','msp','co','naas','iaas','paas','saas','sla','gif','sd-iot','cio','nfv','aas','vmware','wifi','vdi','sdwan.','aws','sd/wan','con','wan.','an','pros','ias','sd_wan','crn','jas','wfh','sap','$wan','bbc','anap','est','hm','hyd','daas','api','am','pm','ans','ipvpn','dr','anap.','idc','nsaas','gbi','att','vpnas','poc','iot','dia','cdw','SCM','scm','ucpaas','ar',"wan'",'ceo','ary','msa','pro','vm50','van','bcp','pubg',"'sd",'cad','laas','sdr','m2m','tuv','iso','cap','s.d','sk','kpn','sdn-wan','waas','arya','vwan','to','vip','mq','tela','ww','www']

        abbr_list = abbr_list + countries_list + countries_abbr
        return abbr_list

    def pre_process(self,keyword, all=True):
        text = re.sub('</?.*?>', ' <> ', str(keyword))
        text = re.sub('(\\d|\\W)+', ' ', str(keyword))
        text = str(keyword).translate(str.maketrans('', '',
                                    string.punctuation))
        # lowercase
        try:
            text = str(text.lower())
        except:
            text = str(text.str.lower())
        text = text.replace("#",'')
        text = text.replace("(",'')
        text = text.replace("()",'')
        tx = []
        for each in text.split(' '):
            tx.append(re.sub('[^a-zA-Z0-9-_*.]', '', each))
        #         tx.append(re.sub('[^A-Za-z0-9]+', ' ', each.lower()))
        text = ' '.join(tx)
        if all:
            # remove tags
            # remove special characters and digits
            if list(filter(None, text.strip())):
                return text.strip()
        else:
            if list(filter(None, text.strip())) \
                and len(text.strip().split(' ')) == 1:
                return text.strip()
    

    def get_drug_names(self,cust):
        drug = pd.read_csv(cust+'_data/brands.csv', header=0)
        drug['BRAND_NAME'] = drug['BRAND_NAME'].apply(self.pre_process)
        drug.dropna(inplace=True)
        drug.drop_duplicates(inplace=True)
        drug = list(set(drug['BRAND_NAME']))
        drug_df = pd.read_csv(cust+'_data/Novartis_drugs.csv')
        drug_df['Brand_Name'] = drug_df['Brand_Name'].apply(self.pre_process)
        drug_df['Drug_Name'] = drug_df['Drug_Name'].apply(self.pre_process)
        drug_df['Brand_Name'] = drug_df['Brand_Name'].apply(lambda x: \
                re.split(r'\s+', x))
        drug_df['Drug_Name'] = drug_df['Drug_Name'].apply(lambda x: \
                re.split(r'\s+', x))
        drug_brand_name = Counter(list(set(sum(drug_df['Brand_Name'
                                ].values.flatten(), []))))
        drug_name = Counter(list(set(sum(drug_df['Drug_Name'].values.flatten(),
                            []))))
        all_drugs_novartis = copy.deepcopy(drug_name)
        all_drugs_novartis.update(drug_brand_name)
        all_drugs = drug + list(set(all_drugs_novartis.keys()))
        all_drugs = Counter(list(set(all_drugs)))
        compititor_drug = set(list(all_drugs.keys())) - set(list(all_drugs_novartis.keys()))
        compititor_drugs = Counter(compititor_drug)
        all_drugs.update(drug)
        all_drugs.update(compititor_drugs)
        all_drugs['ｘｉｉｄｒａ'] = 1
        all_drugs['e n t r e s t o'] = 1
        all_drugs['sacubitril/valsartan'] = 1
        all_drugs["zofran'"] = 1
        all_drugs['promacta/revolade'] = 1
        all_drugs['akcea-apoa-lrx'] = 1
        all_drugs['p i q r a y'] = 1
        all_drugs['zedra'] = 1
        all_drugs['xiddra'] = 1
        all_drugs['mekisit'] = 1
        all_drugs['alepisib'] = 1
        all_drugs['kisquale'] = 1
        all_drugs['secukinab'] = 1
        all_drugs['dabrafbub'] = 1
        all_drugs['dabrafenib'] = 1
        all_drugs['dabrafen'] = 1
        all_drugs['dabraf'] = 1
        all_drugs['makinist'] = 1
        all_drugs['mekanis'] = 1
        all_drugs['mekints'] = 1
        all_drugs['piqray'] = 1
        drug_list = [drug for drug in list(all_drugs.keys()) if len(drug) > 3]
        drug_list = sorted(drug_list, key=len,reverse=True)
        return drug_list

    def get_aryk_list(self,cust):
        BOD = pd.read_csv(cust+'_data/BOD.csv', header=0)
        # BOD['NAME'] = BOD['NAME'].apply(self.pre_process)
        BOD.dropna(inplace=True)
        BOD.drop_duplicates(inplace=True)
        BOD = list(set(BOD['NAME']))

        Imp_words = pd.read_csv(cust+'_data/IMP_list.csv', header=0)
        # BOD['NAME'] = Imp_words['NAME'].apply(self.pre_process)
        Imp_words.dropna(inplace=True)
        Imp_words.drop_duplicates(inplace=True)
        Imp_words = list(set(Imp_words['NAME']))

        Imp_words = Imp_words + BOD
        return Imp_words

# function to run sql query
    def fetch_all_sql(self, query):
        for i in range(3):
            try:
                self.cur = self.con.cursor()
                self.cur.execute(query)
                (query_data, query_decription) = (
                    self.cur.fetchall(),
                    self.cur.description,
                )
                logging.info("Snoflake Query executed successfully!!")
                print('Snoflake Query executed successfully!!')
                return pd.DataFrame(
                    query_data, columns=[col[0] for col in query_decription]
                )
                break
            except snowflake.connector.errors.ProgrammingError as e:
                logging.error("Error while running the query, ERROR: {}".format(e))
            finally:
                self.cur.close()


# Update the intent_lookup table and create a backup table in backup_db 
    def update_intent_table(self):
        logging.info("Updating the Language Translations in Intent LookUp")
        logging.info("TRANSLATION CALLS #: %s" % self.translation_calls)
        try:
            import time

            temp_table = str(time.time()).replace(".", "")
            stmt1 = """ create or replace TABLE {0}.{1}.{2}_{3}(
                        QUERY VARCHAR(4000),
                        QUERY_LANGUAGE_CODE VARCHAR(128),
                        TRANSLATED_QUERY VARCHAR(4000),
                        QUERY_LANGUAGE VARCHAR(128),
                        GCP_TRANSLATED_QUERY VARCHAR(4000),
                        GCP_QUERY_LANGUAGE VARCHAR(128),
                        GCP_QUERY_LANGUAGE_CODE VARCHAR(12))""".format(
                self.BKP_DEPLOY, self.SCHEMA, self.INTENT_TABLE, temp_table
            )
            self.cur = self.con.cursor()
            self.cur.execute(stmt1)
            list_up = []
            logging.info(
                self.keyword_df[
                    (~self.keyword_df["TRANSLATED_QUERY"].isna())
                    & (~self.keyword_df["QUERY_LANGUAGE_CODE"].isna())
                ].count()
            )
            for num1, series in self.keyword_df[
                (~self.keyword_df["TRANSLATED_QUERY"].isna())
                & (~self.keyword_df["QUERY_LANGUAGE_CODE"].isna())
            ].iterrows():
                list_up.append(
                    [
                        series.at["QUERY_LANGUAGE"],
                        series.at["QUERY_LANGUAGE_CODE"],
                        series.at["TRANSLATED_QUERY"].strip(),
                        series.at["QUERY"],
                        series.at['GCP_TRANSLATED_QUERY'],
                        series.at["GCP_QUERY_LANGUAGE_CODE"],
                        series.at["GCP_QUERY_LANGUAGE"]
                    ]
                )
            translation_table = "{0}.{1}.{2}_{3}".format(
                self.BKP_DEPLOY, self.SCHEMA, self.INTENT_TABLE, temp_table
            )
            num = 1600
            s = 0
            e = num
            for i in range(math.ceil(len(list_up) / num)):
                stmt1 = """INSERT INTO {0} (QUERY_LANGUAGE, QUERY_LANGUAGE_CODE, TRANSLATED_QUERY, QUERY
                            , GCP_TRANSLATED_QUERY, GCP_QUERY_LANGUAGE_CODE, GCP_QUERY_LANGUAGE) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s)""".format(
                    translation_table
                )
                self.cur.executemany(stmt1, list_up[s:e])
                s = e
                e = e + num
                print(s)
        except Exception as e:
            logging.error(
                "Error while loading new translation in the database, ERROR: {}".format(
                    e
                )
            )
        try:
            translation_table = "{0}.{1}.{2}".format(
                self.DEPLOY_ENVIRONMENT, self.SCHEMA, self.INTENT_TABLE
            )
            base_table = "{0}.{1}.{2}_{3}".format(
                self.BKP_DEPLOY, self.SCHEMA, self.INTENT_TABLE, temp_table
            )
            self.temp_table = base_table
            stmt1 = """merge into 
                     {0} target_table using {1}
                      source_table 
                    on target_table.QUERY = source_table.QUERY
                       when matched then update 
                       set target_table.QUERY_LANGUAGE_CODE = source_table.QUERY_LANGUAGE_CODE 
                       ,target_table.TRANSLATED_QUERY = trim(source_table.TRANSLATED_QUERY)
                       ,target_table.QUERY_LANGUAGE = source_table.QUERY_LANGUAGE
                    """.format(
                translation_table, base_table
            )
            logging.info(stmt1)
            self.cur.execute(stmt1)
            logging.info("UPDATED THE TRANSLATIONS FOR %s" % self.CUSTOMER)
            logging.info("Temporary Table for language translations: %s" % base_table)
            if (os.environ.get("TEMP_TABLE_CLEANUP")) == "True":
                self.cur.execute("DROP TABLE {0}".format(base_table))
        except Exception as e:
            logging.error(
                "Error while updating the new translated query in intent lookup table in the database, ERROR: {}".format(
                    e
                )
            )
        finally:
            self.cur.close()


# function to get new keywords whose translated_query is null 
    def load_new_keywords(self):
        # Query Lanaguge field in intent lookup
        logging.info(self.NEW_KEYWORDS)
        self.keyword_df = self.fetch_all_sql(self.NEW_KEYWORDS)
        logging.info("Loaded rows: %s" % self.keyword_df.count())
        self.get_keyword_language()

# fuction to call fasttext 
    def fasttext_lang_detect(self, row):
        if (
            not (pd.isnull(row["QUERY_LANGUAGE_CODE"]))
            and row["QUERY_LANGUAGE_CODE"] != ""
        ):
            return row["QUERY_LANGUAGE_CODE"]
        else:
            try:
                detected_languages = self.model.predict(row["QUERY"], k=1)[0][
                    0
                ].replace("__label__", "")
            except Exception as error:
                logging.error("fn: fasttext_lang_detect")
                logging.error(error)
                detected_languages = "un"
            return detected_languages

# function to call polyglot.detector for language detection
    def polyglot_language_detector(self, row):
        """
        Returns the multiple languages with confidence value.For now it's only top 1
        """
        try:
            if (
                not (pd.isnull(row["QUERY_LANGUAGE_CODE"]))
                and row["QUERY_LANGUAGE_CODE"] != ""
            ):
                return row["QUERY_LANGUAGE"]
            detected_languages = Detector(row["QUERY"]).language.name
            logging.info(detected_languages)
        except Exception as error:
            logging.info("LANGUAGE DETECTION ERROR %s" % error)
            detected_languages = "Unassigned"
        return detected_languages

# to get the translation using google api
    def cloud_translation(self, query):
        resp = requests.get(self.API_ENDPOINT.format(query, self.API_KEY))
        return resp

# to detect the language using polyglot detector
    def get_keyword_language(self):
        cnt = self.keyword_df.QUERY.count()
        logging.info("IN GET QUERY LANGUAGE #: %d" % self.keyword_df.QUERY.count())
        if cnt > 0:
            self.keyword_df["QUERY_LANGUAGE"] = self.keyword_df.apply(
                self.polyglot_language_detector, axis=1
            )
            self.keyword_df = pd.merge(
                self.keyword_df,
                self.country_codes,
                left_on="QUERY_LANGUAGE",
                right_on="Language",
                how="left",
            )
            logging.info(
                "POST PANDAS MERGE IN GET QUERY LANGUAGE #: %s"
                % self.keyword_df.QUERY.count()
            )
            self.keyword_df[
                "QUERY_LANGUAGE_CODE"
            ] = self.keyword_df.QUERY_LANGUAGE_CODE.fillna(self.keyword_df.LanguageCode)
            self.keyword_df[
                "QUERY_LANGUAGE_CODE"
            ] = self.keyword_df.QUERY_LANGUAGE_CODE.fillna("un")
            self.keyword_df.drop_duplicates(["QUERY"], keep="last", inplace=True)
            logging.info(
                "QUERY LANGUAGE DETECTION #: %s" % self.keyword_df.QUERY.count()
            )
        else:
            logging.info("No new records for language detection")


# to strip all special characters
    def validate_digits_special_charaters(self, query):
        string_match = list(
            filter(str.strip, (list(re.findall(r"[0-9(\s+*&^%$#@!\-\/\\\+)]+", query))))
        )
        if string_match:
            if set(string_match[0].split()) == set(
                query.strip("\u202c").strip("\u202d").split()
            ):
                return False
            else:
                return True
        else:
            return True

#edited by subhayan 19_08_2021 for custoer independent ranslation
    def get_translation_independent(self, row):
        self.keyword_df.drop_duplicates(["QUERY"], keep="last", inplace=True)
# english translated word
        if row["QUERY_LANGUAGE_CODE"] == "en":
            self.passby_translation_calls = self.passby_translation_calls + 1
            return pd.Series(
                {
                    "TRANSLATED_QUERY": row["QUERY"],
                    "QUERY_LANGUAGE_CODE": row["QUERY_LANGUAGE_CODE"],
                    "QUERY_LANGUAGE": row["QUERY_LANGUAGE"],
                    "GCP_TRANSLATED_QUERY": None,
                    "GCP_QUERY_LANGUAGE_CODE": None,
                    "GCP_QUERY_LANGUAGE": None,
                }
            )
#word without detected language
        elif row["TRANSLATED_QUERY"] is not None:
            self.passby_translation_calls = self.passby_translation_calls + 1
            return pd.Series(
                {
                    "TRANSLATED_QUERY": row["TRANSLATED_QUERY"],
                    "QUERY_LANGUAGE_CODE": row["QUERY_LANGUAGE_CODE"],
                    "QUERY_LANGUAGE": row["QUERY_LANGUAGE"],
                    "GCP_TRANSLATED_QUERY": row["GCP_TRANSLATED_QUERY"],
                    "GCP_QUERY_LANGUAGE_CODE": row["GCP_QUERY_LANGUAGE_CODE"],
                    "GCP_QUERY_LANGUAGE": row["GCP_QUERY_LANGUAGE"],
                }
            )
        else:
            try:
                if (
                        self.validate_digits_special_charaters(row["QUERY"].strip("\u202c"))
                        and len(row["QUERY"]) < 150
                    ): 
                        com_present_flag = 0
                        com_w_comma_present_flag = 0
                        xom_present_flag = 0
                        ca_present_flag = 0
                        om_present_flag = 0
                        com_word_present_flag = 0
                        xom_word_present_flag = 0
                        abbr_present_flag = 0
                        carte_flag = 0
                        com_index = None
                        com_w_comma_index = None
                        xom_index = None
                        ca_index = None
                        om_index = None
                        word_before_com = ''
                        word_before_com_w_comma = ''
                        word_before_xom = ''
                        word_before_ca = ''
                        word_before_om = ''
                        inp_keyword = row["QUERY"]
                        abbr_list = self.get_abbr_list()
                        abbr_dict = dict()
                        com_word_dict = dict()
                        xom_word_dict = dict()
                        row_split = ((str(row["QUERY"])).lower()).split(" ")
                        row_split_dot = ((str(row["QUERY"])).lower()).split(".")
                        row_split_underscore = ((str(row["QUERY"])).lower()).split("_")
# this section check any abbr
                        for i in range(len(row_split)):
                            if re.sub('[,\!\+\#\[\(\)\]\"?]', '', row_split[i]) in abbr_list:
                                abbr_present_flag = 1
                                abbr_dict[row_split.index(row_split[i])] = row_split[i]

# this section is handle com, xom, .xom, .com, .com/ etc.

                        if "com" in row_split:
                            com_word_present_flag = 1
                            com_word_dict[row_split.index("com")] = "com"
                        if "xom" in row_split:
                            xom_word_present_flag = 1
                            xom_word_dict[row_split.index("xom")] = "xom"
                        if ".com " in (row["QUERY"]).lower() or ".com/" in (row["QUERY"]).lower() or ((row["QUERY"]).lower()).endswith(".com"):
                            com_present_flag = 1
                            com_index = ((row["QUERY"]).lower()).index(".com")
                            com_words_list = (row["QUERY"]).split(".")
                            if com_words_list[-1] == "com":
                                word_before_com = com_words_list[-2]
                            else:
                                for i in range(len(com_words_list)):
                                    if "com/" in com_words_list[i] or "com " in com_words_list[i]:
                                        word_before_com = com_words_list[i-1]

                            row["QUERY"] = ((row["QUERY"]).lower()).replace(".com",'')

                        if ",com " in (row["QUERY"]).lower() or ",com/" in (row["QUERY"]).lower() or ((row["QUERY"]).lower()).endswith(",com"):
                            com_w_comma_present_flag = 1
                            com_w_comma_index = ((row["QUERY"]).lower()).index(",com")
                            com_w_comma_words_list = (row["QUERY"]).split(",")
                            if com_w_comma_words_list[-1] == "com":
                                word_before_com_w_comma = com_w_comma_words_list[-2]
                            else:
                                for i in range(len(com_w_comma_words_list)):
                                    if "com/" in com_w_comma_words_list[i] or "com " in com_w_comma_words_list[i]:
                                        word_before_com_w_comma = com_w_comma_words_list[i-1]

                            row["QUERY"] = ((row["QUERY"]).lower()).replace(",com",'')

                        if ".xom " in (row["QUERY"]).lower() or ".xom/" in (row["QUERY"]).lower() or ((row["QUERY"]).lower()).endswith(".xom"):
                            xom_present_flag = 1
                            xom_index = ((row["QUERY"]).lower()).index(".xom")
                            xom_words_list = (row["QUERY"]).split(".")
                            if xom_words_list[-1] == "xom":
                                word_before_xom = xom_words_list[-2]
                            else:
                                for i in range(len(xom_words_list)):
                                    if "xom/" in xom_words_list[i] or "xom " in xom_words_list[i]:
                                        word_before_xom = xom_words_list[i-1]

                            row["QUERY"] = ((row["QUERY"]).lower()).replace(".xom",'')
                        
                        if ".ca " in (row["QUERY"]).lower() or ".ca/" in (row["QUERY"]).lower() or ((row["QUERY"]).lower()).endswith(".ca"):
                            ca_present_flag = 1
                            ca_index = ((row["QUERY"]).lower()).index(".ca")
                            ca_words_list = (row["QUERY"]).split(".")
                            if ca_words_list[-1] == "ca":
                                word_before_ca = ca_words_list[-2]
                            else:
                                for i in range(len(ca_words_list)):
                                    if "ca/" in ca_words_list[i] or "ca " in ca_words_list[i]:
                                        word_before_ca = ca_words_list[i-1]

                            row["QUERY"] = ((row["QUERY"]).lower()).replace(".ca",'')

                        if ".om " in (row["QUERY"]).lower() or ".om/" in (row["QUERY"]).lower() or ((row["QUERY"]).lower()).endswith(".om"):
                            om_present_flag = 1
                            om_index = ((row["QUERY"]).lower()).index(".om")
                            om_words_list = (row["QUERY"]).split(".")
                            if om_words_list[-1] == "ca":
                                word_before_om = om_words_list[-2]
                            else:
                                for i in range(len(om_words_list)):
                                    if "om/" in om_words_list[i] or "om " in om_words_list[i]:
                                        word_before_om = om_words_list[i-1]

                            row["QUERY"] = ((row["QUERY"]).lower()).replace(".om",'')

                            
                        # with open('special_keywords.json', 'r') as f:
                        #         dict_keyword = json.load(f)

                        if self.CUSTOMER in dict_keyword.keys():
                            flag = 0
                            cust_list = dict_keyword[self.CUSTOMER]
                            cust_name = dict()

                            for word in cust_list:
                                if len((row["QUERY"]).split(" ")) == 1 and (row["QUERY"]).lower() in cust_list:
                                    if com_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                    if com_w_comma_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                    if xom_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                    if ca_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                    if abbr_present_flag == 1:
                                        for idx,abbr in abbr_dict.items():
                                                row["QUERY"] = row["QUERY"][:idx] + abbr + row["QUERY"][idx:]
                                    if carte_flag == 1:
                                        row["QUERY"] = row["QUERY"] + " card"

                                    if "&quot;" in row["QUERY"]:
                                        row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                    if "&#39;" in row["QUERY"]:
                                        row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                    return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "en",
                                                    "QUERY_LANGUAGE": "English",
                                                    "GCP_TRANSLATED_QUERY": row["QUERY"],
                                                    "GCP_QUERY_LANGUAGE_CODE": row["QUERY_LANGUAGE_CODE"],
                                                    "GCP_QUERY_LANGUAGE": row["QUERY_LANGUAGE"],
                                                }
                                            )
                                if word in (row["QUERY"]).lower():
                                    flag = 1
                                    index = ((inp_keyword).lower()).index(word)
                                    cust_name[index] = word
                                    row["QUERY"] = ((row["QUERY"]).lower()).replace(word,'')
                                
                            cust_name = collections.OrderedDict(sorted(cust_name.items()))
                            if abbr_present_flag == 1:
                                inp_keyword_split = inp_keyword.split(" ")
                                for abbr in abbr_dict.values():
                                    for i in range(len(inp_keyword_split)):
                                        if inp_keyword_split[i] == abbr:
                                            inp_keyword_split[i] = ''
                                    # inp_keyword = (inp_keyword.lower()).replace(abbr,'')
                                inp_keyword = " ".join(inp_keyword_split)
                            if com_word_present_flag == 1:
                                for com in com_word_dict.values():
                                    inp_keyword = (inp_keyword.lower()).replace(com,'')
                            if xom_word_present_flag == 1:
                                for xom in xom_word_dict.values():
                                    inp_keyword = (inp_keyword.lower()).replace(xom,'')

                            if com_present_flag == 1:
                                inp_keyword = (inp_keyword.lower()).replace(".com",'')
                            if com_w_comma_present_flag == 1:
                                inp_keyword = (inp_keyword.lower()).replace(",com",'')
                            if xom_present_flag == 1:
                                inp_keyword = (inp_keyword.lower()).replace(".xom",'')
                            if ca_present_flag == 1:
                                inp_keyword = (inp_keyword.lower()).replace(".ca",'')
                            if om_present_flag == 1:
                                inp_keyword = (inp_keyword.lower()).replace(".om",'')
                            if "carte" in row_split:
                                carte_flag = 1
                                inp_keyword = (inp_keyword.lower()).replace("carte",'')

                            row["QUERY"] = inp_keyword
                            for word in cust_name.values():
                                row["QUERY"] = ((row["QUERY"]).lower()).replace(word,'')
                            
                            if flag == 1:
                                resp = self.cloud_translation(row["QUERY"])
                                if resp.status_code == 200:
                                        response = [
                                            json.loads(resp.content)["data"]["translations"][0][
                                                "translatedText"
                                            ],
                                            json.loads(resp.content)["data"]["translations"][0][
                                                "detectedSourceLanguage"
                                            ],
                                        ]
                                        self.translation_calls = self.translation_calls + 1
                                        new_lang = (
                                            self.country_codes[
                                                self.country_codes["LanguageCode"] == response[1]
                                            ]["Language"]
                                            .head(1)
                                            .values
                                        )
                                        if new_lang:
                                            language = new_lang[0]
                                        else:
                                            language = "Unassigned"
                                        if response[1] == 'en':
                                            for idx,word in cust_name.items():
                                                row["QUERY"] = row["QUERY"][:idx] + word + row["QUERY"][idx:]
                                            if abbr_present_flag == 1:
                                                row_split_words = (row["QUERY"]).split(" ")
                                                for idx,abbr in abbr_dict.items():
                                                    row_split_words.insert(idx,abbr)
                                                row["QUERY"] = " ".join(row_split_words)
                                            
                                            if com_word_present_flag == 1:
                                                row_split_words = (row["QUERY"]).split(" ")
                                                for idx,com in com_word_dict.items():
                                                    row_split_words.insert(idx,com)
                                                row["QUERY"] = " ".join(row_split_words)
                                            if xom_word_present_flag == 1:
                                                row_split_words = (row["QUERY"]).split(" ")
                                                for idx,xom in xom_word_dict.items():
                                                    row_split_words.insert(idx,xom)
                                                row["QUERY"] = " ".join(row_split_words)

                                            if com_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                            if com_w_comma_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                            if xom_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                            if ca_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                            if om_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:om_index] + ".om" + row["QUERY"][om_index:]
                                            if "&quot;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                            if "&#39;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                            if carte_flag == 1:
                                                row["QUERY"] = row["QUERY"] + " card"
                                                return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": row["QUERY"],
                                                        "QUERY_LANGUAGE_CODE": "fr",
                                                        "QUERY_LANGUAGE": "French",
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                )
                                            else:
                                                if "&quot;" in row["QUERY"]:
                                                    row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                                if "&#39;" in row["QUERY"]:
                                                    row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                                return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": row["QUERY"],
                                                        "QUERY_LANGUAGE_CODE": "en",
                                                        "QUERY_LANGUAGE": "English",
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                )
                                        if row["QUERY"] == response[0] and row["IS_BRANDED"] == True and language != 'en'\
                                            and ((ord(row["QUERY"][0]) in range(0, 123)) 
                                                    or (ord(row["QUERY"][-1]) in range(0, 123))):
                                            for idx,word in cust_name.items():
                                                row["QUERY"] = row["QUERY"][:idx] + word + row["QUERY"][idx:]
                                            if abbr_present_flag == 1:
                                                row_split_words = (row["QUERY"]).split(" ")
                                                for idx,abbr in abbr_dict.items():
                                                    row_split_words.insert(idx,abbr)
                                                row["QUERY"] = " ".join(row_split_words)
                                            
                                            if com_word_present_flag == 1:
                                                row_split_words = (row["QUERY"]).split(" ")
                                                for idx,com in com_word_dict.items():
                                                    row_split_words.insert(idx,com)
                                                row["QUERY"] = " ".join(row_split_words)
                                            if xom_word_present_flag == 1:
                                                row_split_words = (row["QUERY"]).split(" ")
                                                for idx,xom in xom_word_dict.items():
                                                    row_split_words.insert(idx,xom)
                                                row["QUERY"] = " ".join(row_split_words)
                                            
                                            if com_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                            if com_w_comma_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                            if xom_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                            if ca_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                            if om_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:om_index] + ".om" + row["QUERY"][om_index:]
                                            if "&quot;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                            if "&#39;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                            if carte_flag == 1:
                                                row["QUERY"] = row["QUERY"] + " card"
                                                return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": row["QUERY"],
                                                        "QUERY_LANGUAGE_CODE": "fr",
                                                        "QUERY_LANGUAGE": "French",
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                )
                                            else:
                                                if "&quot;" in row["QUERY"]:
                                                    row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                                if "&#39;" in row["QUERY"]:
                                                    row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                                return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": row["QUERY"],
                                                        "QUERY_LANGUAGE_CODE": "en",
                                                        "QUERY_LANGUAGE": "English",
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                )
                                        else:
                                                resp_w_word = response[0]
                                                for word in cust_name.values():
                                                    resp_w_word = resp_w_word + " " + word
                                                if com_present_flag == 1:
                                                    if word_before_com in resp_w_word:
                                                        resp_w_word = resp_w_word.replace(word_before_com,(word_before_com+".com"))
                                                    else:
                                                        resp_w_word = resp_w_word + ".com"
                                                if com_w_comma_present_flag == 1:
                                                    if word_before_com_w_comma in resp_w_word:
                                                        resp_w_word = resp_w_word.replace(word_before_com_w_comma,(word_before_com_w_comma+",com"))
                                                    else:
                                                        resp_w_word = resp_w_word + ",com"
                                                if xom_present_flag == 1:
                                                    if word_before_xom in resp_w_word:
                                                        resp_w_word = resp_w_word.replace(word_before_xom,(word_before_xom+".xom"))
                                                    else:
                                                        resp_w_word = resp_w_word + ".xom"
                                                if ca_present_flag == 1:
                                                    if word_before_ca in resp_w_word:
                                                        resp_w_word = resp_w_word.replace(word_before_ca,(word_before_ca+".ca"))
                                                    else:
                                                        resp_w_word = resp_w_word + ".ca"
                                                if om_present_flag == 1:
                                                    if word_before_om in resp_w_word:
                                                        resp_w_word = resp_w_word.replace(word_before_om,(word_before_om+".om"))
                                                    else:
                                                        resp_w_word = resp_w_word + ".om"

                                                if abbr_present_flag == 1:
                                                    for idx,abbr in abbr_dict.items():
                                                        resp_w_word = resp_w_word + " " + abbr
                                                if com_word_present_flag == 1:
                                                    for idx,com in com_word_dict.items():
                                                        resp_w_word = resp_w_word + " " + com
                                                if xom_word_present_flag == 1:
                                                    for idx,xom in xom_word_dict.items():
                                                        resp_w_word = resp_w_word + " " + xom
                                                if "&quot;" in resp_w_word:
                                                    resp_w_word = (resp_w_word).replace("&quot;",'"')
                                                if "&#39;" in resp_w_word:
                                                    resp_w_word = (resp_w_word).replace("&#39;","'")
                                                if carte_flag == 1:
                                                    resp_w_word = resp_w_word + " card"
                                                    return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": resp_w_word,
                                                        "QUERY_LANGUAGE_CODE": "fr",
                                                        "QUERY_LANGUAGE": "French",
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                    )
                                                else:
                                                    if "&quot;" in resp_w_word:
                                                        resp_w_word = (resp_w_word).replace("&quot;",'"')
                                                    if "&#39;" in resp_w_word:
                                                        resp_w_word = (resp_w_word).replace("&#39;","'")
                                                    return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": resp_w_word,
                                                        "QUERY_LANGUAGE_CODE": response[1],
                                                        "QUERY_LANGUAGE": language,
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                    )
                                else:
                                    self.failed_translation_calls = (
                                            self.failed_translation_calls + 1
                                        )
                                    for idx,word in cust_name.items():
                                        row["QUERY"] = row["QUERY"][:idx] + word + row["QUERY"][idx:]
                                    if abbr_present_flag == 1:
                                        row_split_words = (row["QUERY"]).split(" ")
                                        for idx,abbr in abbr_dict.items():
                                            row_split_words.insert(idx,abbr)
                                        row["QUERY"] = " ".join(row_split_words)
                                    
                                    if com_word_present_flag == 1:
                                        row_split_words = (row["QUERY"]).split(" ")
                                        for idx,com in com_word_dict.items():
                                            row_split_words.insert(idx,com)
                                        row["QUERY"] = " ".join(row_split_words)
                                    if xom_word_present_flag == 1:
                                        row_split_words = (row["QUERY"]).split(" ")
                                        for idx,xom in xom_word_dict.items():
                                            row_split_words.insert(idx,xom)
                                        row["QUERY"] = " ".join(row_split_words)
                                    
                                    if com_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                    if com_w_comma_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                    if xom_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                    if ca_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                    if "&quot;" in row["QUERY"]:
                                        row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                    if "&#39;" in row["QUERY"]:
                                        row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                    if carte_flag == 1:
                                        row["QUERY"] = row["QUERY"] + " card"
                                        return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": row["QUERY"],
                                                        "QUERY_LANGUAGE_CODE": "fr",
                                                        "QUERY_LANGUAGE": "French",
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                )
                                    else:
                                        if "&quot;" in row["QUERY"]:
                                            row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                        if "&#39;" in row["QUERY"]:
                                            row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                        return pd.Series(
                                            {
                                                "TRANSLATED_QUERY": row["QUERY"],
                                                "QUERY_LANGUAGE_CODE": "en",
                                                "QUERY_LANGUAGE": "English",
                                                "GCP_TRANSLATED_QUERY": response[0],
                                                "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                "GCP_QUERY_LANGUAGE": language,
                                            }
                                        )
                            if flag == 0:             #No ARYK imp word present
                                    resp = self.cloud_translation(row["QUERY"])
                                    if resp.status_code == 200:
                                        response = [
                                            json.loads(resp.content)["data"]["translations"][0][
                                                "translatedText"
                                            ],
                                            json.loads(resp.content)["data"]["translations"][0][
                                                "detectedSourceLanguage"
                                            ],
                                        ]
                                        self.translation_calls = self.translation_calls + 1
                                        new_lang = (
                                            self.country_codes[
                                                self.country_codes["LanguageCode"] == response[1]
                                            ]["Language"]
                                            .head(1)
                                            .values
                                        )
                                        if new_lang:
                                            language = new_lang[0]
                                        else:
                                            language = "Unassigned"
                                        if response[1] == 'en':
                                            if abbr_present_flag == 1:
                                                row_split_words = (row["QUERY"]).split(" ")
                                                for idx,abbr in abbr_dict.items():
                                                    row_split_words.insert(idx,abbr)
                                                row["QUERY"] = " ".join(row_split_words)
                                            
                                            if com_word_present_flag == 1:
                                                row_split_words = (row["QUERY"]).split(" ")
                                                for idx,com in com_word_dict.items():
                                                    row_split_words.insert(idx,com)
                                                row["QUERY"] = " ".join(row_split_words)
                                            if xom_word_present_flag == 1:
                                                row_split_words = (row["QUERY"]).split(" ")
                                                for idx,xom in xom_word_dict.items():
                                                    row_split_words.insert(idx,xom)
                                                row["QUERY"] = " ".join(row_split_words)

                                            if com_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                            if com_w_comma_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                            if xom_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                            if ca_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                            if om_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:om_index] + ".om" + row["QUERY"][om_index:]
                                            if "&quot;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                            if "&#39;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                            if carte_flag == 1:
                                                row["QUERY"] = row["QUERY"] + " card"
                                                return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": row["QUERY"],
                                                        "QUERY_LANGUAGE_CODE": "fr",
                                                        "QUERY_LANGUAGE": "French",
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                )
                                            else:
                                                if "&quot;" in row["QUERY"]:
                                                    row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                                if "&#39;" in row["QUERY"]:
                                                    row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                                return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": row["QUERY"],
                                                        "QUERY_LANGUAGE_CODE": "en",
                                                        "QUERY_LANGUAGE": "English",
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                )
                                        if row["QUERY"] == response[0] and row["IS_BRANDED"] == True and language != 'en'\
                                            and ((ord(row["QUERY"][0]) in range(0, 123)) 
                                                    or (ord(row["QUERY"][-1]) in range(0, 123))):
                                            if abbr_present_flag == 1:
                                                row_split_words = (row["QUERY"]).split(" ")
                                                for idx,abbr in abbr_dict.items():
                                                    row_split_words.insert(idx,abbr)
                                                row["QUERY"] = " ".join(row_split_words)
                                            if com_word_present_flag == 1:
                                                row_split_words = (row["QUERY"]).split(" ")
                                                for idx,com in com_word_dict.items():
                                                    row_split_words.insert(idx,com)
                                                row["QUERY"] = " ".join(row_split_words)
                                            if xom_word_present_flag == 1:
                                                row_split_words = (row["QUERY"]).split(" ")
                                                for idx,xom in xom_word_dict.items():
                                                    row_split_words.insert(idx,xom)
                                                row["QUERY"] = " ".join(row_split_words)

                                            if com_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                            if com_w_comma_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                            if xom_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                            if ca_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                            if om_present_flag == 1:
                                                row["QUERY"] = row["QUERY"][:om_index] + ".om" + row["QUERY"][om_index:]
                                            if "&quot;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                            if "&#39;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                            if carte_flag == 1:
                                                row["QUERY"] = row["QUERY"] + " card"
                                                return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": row["QUERY"],
                                                        "QUERY_LANGUAGE_CODE": "fr",
                                                        "QUERY_LANGUAGE": "French",
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                )
                                            else:
                                                if "&quot;" in row["QUERY"]:
                                                    row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                                if "&#39;" in row["QUERY"]:
                                                    row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                                return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": row["QUERY"],
                                                        "QUERY_LANGUAGE_CODE": "en",
                                                        "QUERY_LANGUAGE": "English",
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                )
                                        else:
                                                resp_w_word = response[0]
                                                if com_present_flag == 1:
                                                    if word_before_com in resp_w_word:
                                                        resp_w_word = resp_w_word.replace(word_before_com,(word_before_com+".com"))
                                                    else:
                                                        resp_w_word = resp_w_word + ".com"
                                                if com_w_comma_present_flag == 1:
                                                    if word_before_com_w_comma in resp_w_word:
                                                        resp_w_word = resp_w_word.replace(word_before_com_w_comma,(word_before_com_w_comma+",com"))
                                                    else:
                                                        resp_w_word = resp_w_word + ",com"
                                                if xom_present_flag == 1:
                                                    if word_before_xom in resp_w_word:
                                                        resp_w_word = resp_w_word.replace(word_before_xom,(word_before_xom+".xom"))
                                                    else:
                                                        resp_w_word = resp_w_word + ".xom"
                                                if ca_present_flag == 1:
                                                    if word_before_ca in resp_w_word:
                                                        resp_w_word = resp_w_word.replace(word_before_ca,(word_before_ca+".ca"))
                                                    else:
                                                        resp_w_word = resp_w_word + ".ca"
                                                if om_present_flag == 1:
                                                    if word_before_om in resp_w_word:
                                                        resp_w_word = resp_w_word.replace(word_before_om,(word_before_om+".om"))
                                                    else:
                                                        resp_w_word = resp_w_word + ".om"
                                                if abbr_present_flag == 1:
                                                    for idx,abbr in abbr_dict.items():
                                                        resp_w_word = resp_w_word + " " + abbr

                                                if com_word_present_flag == 1:
                                                    for idx,com in com_word_dict.items():
                                                        resp_w_word = resp_w_word + " " + com
                                                if xom_word_present_flag == 1:
                                                    for idx,xom in xom_word_dict.items():
                                                        resp_w_word = resp_w_word + " " + xom
                                                if "&quot;" in resp_w_word:
                                                    resp_w_word = (resp_w_word).replace("&quot;",'"')
                                                if "&#39;" in resp_w_word:
                                                    resp_w_word = (resp_w_word).replace("&#39;","'")
                                                if carte_flag == 1:
                                                    resp_w_word = resp_w_word + " card"
                                                    return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": resp_w_word,
                                                        "QUERY_LANGUAGE_CODE": "fr",
                                                        "QUERY_LANGUAGE": "French",
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                    )
                                                else:
                                                    if "&quot;" in resp_w_word:
                                                        resp_w_word = (resp_w_word).replace("&quot;",'"')
                                                    if "&#39;" in resp_w_word:
                                                        resp_w_word = (resp_w_word).replace("&#39;","'")
                                                    return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": resp_w_word,
                                                        "QUERY_LANGUAGE_CODE": response[1],
                                                        "QUERY_LANGUAGE": language,
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                    )
                                    else:
                                        self.failed_translation_calls = (
                                            self.failed_translation_calls + 1
                                        )
                                        if abbr_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,abbr in abbr_dict.items():
                                                row_split_words.insert(idx,abbr)
                                            row["QUERY"] = " ".join(row_split_words)
                                        if com_word_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,com in com_word_dict.items():
                                                row_split_words.insert(idx,com)
                                            row["QUERY"] = " ".join(row_split_words)
                                        if xom_word_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,xom in xom_word_dict.items():
                                                row_split_words.insert(idx,xom)
                                            row["QUERY"] = " ".join(row_split_words)

                                        if com_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                        if com_w_comma_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                        if xom_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                        if ca_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                        if om_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:om_index] + ".om" + row["QUERY"][om_index:]
                                        if "&quot;" in row["QUERY"]:
                                            row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                        if "&#39;" in row["QUERY"]:
                                            row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                        if carte_flag == 1:
                                            row["QUERY"] = row["QUERY"] + " card"
                                            return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": row["QUERY"],
                                                        "QUERY_LANGUAGE_CODE": "fr",
                                                        "QUERY_LANGUAGE": "French",
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                )
                                        else:
                                                if "&quot;" in row["QUERY"]:
                                                    row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                                if "&#39;" in row["QUERY"]:
                                                    row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                                return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": row["QUERY"],
                                                        "QUERY_LANGUAGE_CODE": "en",
                                                        "QUERY_LANGUAGE": "English",
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                )
                        #  this is for others customer#  this is for others customer
                        else:
                            resp = self.cloud_translation(row["QUERY"])
                            if resp.status_code == 200:
                                response = [
                                    json.loads(resp.content)["data"]["translations"][0][
                                        "translatedText"
                                    ],
                                    json.loads(resp.content)["data"]["translations"][0][
                                        "detectedSourceLanguage"
                                    ],
                                ]
                                self.translation_calls = self.translation_calls + 1
                                new_lang = (
                                    self.country_codes[
                                        self.country_codes["LanguageCode"] == response[1]
                                    ]["Language"]
                                    .head(1)
                                    .values
                                )
                                if new_lang:
                                    language = new_lang[0]
                                else:
                                    language = "Unassigned"

                                if row["QUERY"] == response[0] and row["IS_BRANDED"] == True and language != 'en'\
                                    and ((ord(row["QUERY"][0]) in range(0, 123)) 
                                            or (ord(row["QUERY"][-1]) in range(0, 123))):
                                    if abbr_present_flag == 1:
                                        row_split_words = (row["QUERY"]).split(" ")
                                        for idx,abbr in abbr_dict.items():
                                            row_split_words.insert(idx,abbr)
                                        row["QUERY"] = " ".join(row_split_words)
                                    if com_word_present_flag == 1:
                                                row_split_words = (row["QUERY"]).split(" ")
                                                for idx,com in com_word_dict.items():
                                                    row_split_words.insert(idx,com)
                                                row["QUERY"] = " ".join(row_split_words)
                                    if xom_word_present_flag == 1:
                                        row_split_words = (row["QUERY"]).split(" ")
                                        for idx,xom in xom_word_dict.items():
                                            row_split_words.insert(idx,xom)
                                        row["QUERY"] = " ".join(row_split_words)

                                    if com_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                    if com_w_comma_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                    if xom_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                    if ca_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                    if om_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:om_index] + ".om" + row["QUERY"][om_index:]
                                    if "&quot;" in row["QUERY"]:
                                        row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                    if "&#39;" in row["QUERY"]:
                                        row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                    if carte_flag == 1:
                                        row["QUERY"] = row["QUERY"] + " card"
                                        return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": row["QUERY"],
                                                        "QUERY_LANGUAGE_CODE": "fr",
                                                        "QUERY_LANGUAGE": "French",
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                )


                                    else:
                                                if "&quot;" in row["QUERY"]:
                                                    row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                                if "&#39;" in row["QUERY"]:
                                                    row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                                return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": row["QUERY"],
                                                        "QUERY_LANGUAGE_CODE": "en",
                                                        "QUERY_LANGUAGE": "English",
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                )

                                else:
                                        resp_w_drug = response[0]
                                        if com_present_flag == 1:
                                            if word_before_com in resp_w_drug:
                                                resp_w_drug = resp_w_drug.replace(word_before_com,(word_before_com+".com"))
                                            else:
                                                resp_w_drug = resp_w_drug + ".com"
                                        if com_w_comma_present_flag == 1:
                                            if word_before_com_w_comma in resp_w_drug:
                                                resp_w_drug = resp_w_drug.replace(word_before_com_w_comma,(word_before_com_w_comma+",com"))
                                            else:
                                                resp_w_drug = resp_w_drug + ",com"
                                        if xom_present_flag == 1:
                                            if word_before_xom in resp_w_drug:
                                                resp_w_drug = resp_w_drug.replace(word_before_xom,(word_before_xom+".xom"))
                                            else:
                                                resp_w_drug = resp_w_drug + ".xom"
                                        if ca_present_flag == 1:
                                            if word_before_ca in resp_w_drug:
                                                resp_w_drug = resp_w_drug.replace(word_before_ca,(word_before_ca+".ca"))
                                            else:
                                                resp_w_drug = resp_w_drug + ".ca"
                                        if om_present_flag == 1:
                                            if word_before_om in resp_w_drug:
                                                resp_w_drug = resp_w_drug.replace(word_before_om,(word_before_om+".om"))
                                            else:
                                                resp_w_drug = resp_w_drug + ".om"
                                        if abbr_present_flag == 1:
                                            for idx,abbr in abbr_dict.items():
                                                resp_w_drug = resp_w_drug + " " +abbr
                                        
                                        if com_word_present_flag == 1:
                                            for idx,com in com_word_dict.items():
                                                resp_w_drug = resp_w_drug + " " + com
                                        if xom_word_present_flag == 1:
                                            for idx,xom in xom_word_dict.items():
                                                resp_w_drug = resp_w_drug + " " + xom
                                        if "&quot;" in resp_w_drug:
                                            resp_w_drug = (resp_w_drug).replace("&quot;",'"')
                                        if "&#39;" in resp_w_drug:
                                            resp_w_drug = (resp_w_drug).replace("&#39;","'")
                                        if carte_flag == 1:
                                            resp_w_drug = resp_w_drug + " card"
                                            return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": resp_w_drug,
                                                        "QUERY_LANGUAGE_CODE": "fr",
                                                        "QUERY_LANGUAGE": "French",
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                    )
                                        else:
                                                    if "&quot;" in resp_w_drug:
                                                        resp_w_drug = (resp_w_drug).replace("&quot;",'"')
                                                    if "&#39;" in resp_w_drug:
                                                        resp_w_drug = (resp_w_drug).replace("&#39;","'")
                                                    return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": resp_w_drug,
                                                        "QUERY_LANGUAGE_CODE": response[1],
                                                        "QUERY_LANGUAGE": language,
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                    )
                            else:
                                self.failed_translation_calls = (
                                    self.failed_translation_calls + 1
                                )
                                if abbr_present_flag == 1:
                                    row_split_words = (row["QUERY"]).split(" ")
                                    for idx,abbr in abbr_dict.items():
                                        row_split_words.insert(idx,abbr)
                                    row["QUERY"] = " ".join(row_split_words)
                                if com_word_present_flag == 1:
                                    row_split_words = (row["QUERY"]).split(" ")
                                    for idx,com in com_word_dict.items():
                                        row_split_words.insert(idx,com)
                                    row["QUERY"] = " ".join(row_split_words)
                                if xom_word_present_flag == 1:
                                    row_split_words = (row["QUERY"]).split(" ")
                                    for idx,xom in xom_word_dict.items():
                                        row_split_words.insert(idx,xom)
                                    row["QUERY"] = " ".join(row_split_words)

                                if com_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                if com_w_comma_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                if xom_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                if ca_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                if om_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:om_index] + ".om" + row["QUERY"][om_index:]
                                if "&quot;" in row["QUERY"]:
                                    row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                if "&#39;" in row["QUERY"]:
                                    row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                if carte_flag == 1:
                                    row["QUERY"] = row["QUERY"] + " card"
                                    return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": row["QUERY"],
                                                        "QUERY_LANGUAGE_CODE": "fr",
                                                        "QUERY_LANGUAGE": "French",
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                )
                                else:
                                                if "&quot;" in row["QUERY"]:
                                                    row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                                if "&#39;" in row["QUERY"]:
                                                    row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                                return pd.Series(
                                                    {
                                                        "TRANSLATED_QUERY": row["QUERY"],
                                                        "QUERY_LANGUAGE_CODE": "en",
                                                        "QUERY_LANGUAGE": "English",
                                                        "GCP_TRANSLATED_QUERY": response[0],
                                                        "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                        "GCP_QUERY_LANGUAGE": language,
                                                    }
                                                )
                    
        # this section is for passby translation                
                else:
                    self.passby_translation_calls = self.passby_translation_calls + 1
                    if "&quot;" in row["QUERY"]:
                        row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                    if "&#39;" in row["QUERY"]:
                        row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                    return pd.Series(
                        {
                            "TRANSLATED_QUERY": row["QUERY"],
                            "QUERY_LANGUAGE_CODE": row["QUERY_LANGUAGE_CODE"],
                            "QUERY_LANGUAGE": row["QUERY_LANGUAGE"],
                            "GCP_TRANSLATED_QUERY": None,
                            "GCP_QUERY_LANGUAGE_CODE": None,
                            "GCP_QUERY_LANGUAGE": None,
                        }
                    )
                    
        # this section is for failed translation       
            except Exception as e:
                self.failed_translation_calls = self.failed_translation_calls + 1
                logging.error("Translation Error %s" % (e))
                self.update_intent_table()
                if (self.translation_calls / self.BATCH_SIZE).is_integer():
                    logging.info(self.translation_calls)
                return pd.Series(
                    {
                        "TRANSLATED_QUERY": None,
                        "QUERY_LANGUAGE_CODE": row["QUERY_LANGUAGE_CODE"],
                        "QUERY_LANGUAGE": row["QUERY_LANGUAGE"],
                        "GCP_TRANSLATED_QUERY": None,
                        "GCP_QUERY_LANGUAGE_CODE": None,
                        "GCP_QUERY_LANGUAGE": None,
                    }
                )


#transation starts
    def get_keyword_translation(self, row):
        self.keyword_df.drop_duplicates(["QUERY"], keep="last", inplace=True)
# for those keywords whose language is English
        if row["QUERY_LANGUAGE_CODE"] == "en":
            self.passby_translation_calls = self.passby_translation_calls + 1
            return pd.Series(
                {
                    "TRANSLATED_QUERY": row["QUERY"],
                    "QUERY_LANGUAGE_CODE": row["QUERY_LANGUAGE_CODE"],
                    "QUERY_LANGUAGE": row["QUERY_LANGUAGE"],
                    "GCP_TRANSLATED_QUERY": None,
                    "GCP_QUERY_LANGUAGE_CODE": None,
                    "GCP_QUERY_LANGUAGE": None,
                }
            )
# for those keywords whose translation was not done by polyglot
        elif row["TRANSLATED_QUERY"] is not None:
            self.passby_translation_calls = self.passby_translation_calls + 1
            return pd.Series(
                {
                    "TRANSLATED_QUERY": row["TRANSLATED_QUERY"],
                    "QUERY_LANGUAGE_CODE": row["QUERY_LANGUAGE_CODE"],
                    "QUERY_LANGUAGE": row["QUERY_LANGUAGE"],
                    "GCP_TRANSLATED_QUERY": row["GCP_TRANSLATED_QUERY"],
                    "GCP_QUERY_LANGUAGE_CODE": row["GCP_QUERY_LANGUAGE_CODE"],
                    "GCP_QUERY_LANGUAGE": row["GCP_QUERY_LANGUAGE"],
                }
            )

#from here code is for all the keywords with language as non-english
        else:
            try:

# initialisation for translation job
                if (
                    self.validate_digits_special_charaters(row["QUERY"].strip("\u202c"))
                    and len(row["QUERY"]) < 150
                ):
                    com_present_flag = 0
                    com_w_comma_present_flag = 0
                    xom_present_flag = 0
                    ca_present_flag = 0
                    om_present_flag = 0
                    com_word_present_flag = 0
                    xom_word_present_flag = 0
                    abbr_present_flag = 0
                    carte_flag = 0
                    com_index = None
                    com_w_comma_index = None
                    xom_index = None
                    ca_index = None
                    om_index = None
                    word_before_com = ''
                    word_before_com_w_comma = ''
                    word_before_xom = ''
                    word_before_ca = ''
                    word_before_om = ''
                    inp_keyword = row["QUERY"]
                    abbr_list = self.get_abbr_list()
                    abbr_dict = dict()
                    com_word_dict = dict()
                    xom_word_dict = dict()
                    row_split = ((str(row["QUERY"])).lower()).split(" ")
                    row_split_dot = ((str(row["QUERY"])).lower()).split(".")
                    row_split_underscore = ((str(row["QUERY"])).lower()).split("_")
# this section check any abbr
                    for i in range(len(row_split)):
                        if re.sub('[,\!\+\#\[\(\)\]\"?]', '', row_split[i]) in abbr_list:
                            abbr_present_flag = 1
                            abbr_dict[row_split.index(row_split[i])] = row_split[i]

# this section is handle com, xom, .xom, .com, .com/ etc.

                    if "com" in row_split:
                        com_word_present_flag = 1
                        com_word_dict[row_split.index("com")] = "com"
                    if "xom" in row_split:
                        xom_word_present_flag = 1
                        xom_word_dict[row_split.index("xom")] = "xom"

                    if ".com " in (row["QUERY"]).lower() or ".com/" in (row["QUERY"]).lower() or ((row["QUERY"]).lower()).endswith(".com"):
                        com_present_flag = 1
                        com_index = ((row["QUERY"]).lower()).index(".com")
                        com_words_list = (row["QUERY"]).split(".")
                        if com_words_list[-1] == "com":
                            word_before_com = com_words_list[-2]
                        else:
                            for i in range(len(com_words_list)):
                                if "com/" in com_words_list[i] or "com " in com_words_list[i]:
                                    word_before_com = com_words_list[i-1]

                        row["QUERY"] = ((row["QUERY"]).lower()).replace(".com",'')

                    if ",com " in (row["QUERY"]).lower() or ",com/" in (row["QUERY"]).lower() or ((row["QUERY"]).lower()).endswith(",com"):
                        com_w_comma_present_flag = 1
                        com_w_comma_index = ((row["QUERY"]).lower()).index(",com")
                        com_w_comma_words_list = (row["QUERY"]).split(",")
                        if com_w_comma_words_list[-1] == "com":
                            word_before_com_w_comma = com_w_comma_words_list[-2]
                        else:
                            for i in range(len(com_w_comma_words_list)):
                                if "com/" in com_w_comma_words_list[i] or "com " in com_w_comma_words_list[i]:
                                    word_before_com_w_comma = com_w_comma_words_list[i-1]

                        row["QUERY"] = ((row["QUERY"]).lower()).replace(",com",'')

                    if ".xom " in (row["QUERY"]).lower() or ".xom/" in (row["QUERY"]).lower() or ((row["QUERY"]).lower()).endswith(".xom"):
                        xom_present_flag = 1
                        xom_index = ((row["QUERY"]).lower()).index(".xom")
                        xom_words_list = (row["QUERY"]).split(".")
                        if xom_words_list[-1] == "xom":
                            word_before_xom = xom_words_list[-2]
                        else:
                            for i in range(len(xom_words_list)):
                                if "xom/" in xom_words_list[i] or "xom " in xom_words_list[i]:
                                    word_before_xom = xom_words_list[i-1]

                        row["QUERY"] = ((row["QUERY"]).lower()).replace(".xom",'')

                    if ".ca " in (row["QUERY"]).lower() or ".ca/" in (row["QUERY"]).lower() or ((row["QUERY"]).lower()).endswith(".ca"):
                        ca_present_flag = 1
                        ca_index = ((row["QUERY"]).lower()).index(".ca")
                        ca_words_list = (row["QUERY"]).split(".")
                        if ca_words_list[-1] == "ca":
                            word_before_ca = ca_words_list[-2]
                        else:
                            for i in range(len(ca_words_list)):
                                if "ca/" in ca_words_list[i] or "ca " in ca_words_list[i]:
                                    word_before_ca = ca_words_list[i-1]

                        row["QUERY"] = ((row["QUERY"]).lower()).replace(".ca",'')

                    if ".om " in (row["QUERY"]).lower() or ".om/" in (row["QUERY"]).lower() or ((row["QUERY"]).lower()).endswith(".om"):
                        om_present_flag = 1
                        om_index = ((row["QUERY"]).lower()).index(".om")
                        om_words_list = (row["QUERY"]).split(".")
                        if om_words_list[-1] == "ca":
                            word_before_om = om_words_list[-2]
                        else:
                            for i in range(len(om_words_list)):
                                if "om/" in om_words_list[i] or "om " in om_words_list[i]:
                                    word_before_om = om_words_list[i-1]

                        row["QUERY"] = ((row["QUERY"]).lower()).replace(".om",'')

#  this is the part where NVSO, NVSP and NVS is taken care of
                    

                    if self.CUSTOMER == 'NVSP' or self.CUSTOMER == 'NVSO' or self.CUSTOMER == 'NVS':
                        flag = 0
                        all_drugs = self.get_drug_names(self.CUSTOMER)
                        drug_names = dict()
                        for drug in all_drugs:
                            if len((row["QUERY"]).split(" ")) == 1 and (row["QUERY"]).lower() in all_drugs:
                                if com_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                if com_w_comma_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                if xom_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                if ca_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                if abbr_present_flag == 1:
                                    for idx,abbr in abbr_dict.items():
                                            row["QUERY"] = row["QUERY"][:idx] + abbr + row["QUERY"][idx:]
                                if carte_flag == 1:
                                    row["QUERY"] = row["QUERY"] + " card"
                                return pd.Series(
                                            {
                                                "TRANSLATED_QUERY": row["QUERY"],
                                                "QUERY_LANGUAGE_CODE": "en",
                                                "QUERY_LANGUAGE": "English",
                                                "GCP_TRANSLATED_QUERY": row["QUERY"],
                                                "GCP_QUERY_LANGUAGE_CODE": row["QUERY_LANGUAGE_CODE"],
                                                "GCP_QUERY_LANGUAGE": row["QUERY_LANGUAGE"],
                                            }
                                        )
                            if drug in (row["QUERY"]).lower():
                                flag = 1
                                index = ((inp_keyword).lower()).index(drug)
                                drug_names[index] = drug
                                row["QUERY"] = ((row["QUERY"]).lower()).replace(drug,'')

                        drug_names = collections.OrderedDict(sorted(drug_names.items()))
                        if abbr_present_flag == 1:
                            for abbr in abbr_dict.values():
                                inp_keyword = (inp_keyword.lower()).replace(abbr,'')
                        if com_word_present_flag == 1:
                            for com in com_word_dict.values():
                                inp_keyword = (inp_keyword.lower()).replace(com,'')
                        if xom_word_present_flag == 1:
                            for xom in xom_word_dict.values():
                                inp_keyword = (inp_keyword.lower()).replace(xom,'')

                        if com_present_flag == 1:
                            inp_keyword = (inp_keyword.lower()).replace(".com",'')
                        if com_w_comma_present_flag == 1:
                            inp_keyword = (inp_keyword.lower()).replace(",com",'')
                        if xom_present_flag == 1:
                            inp_keyword = (inp_keyword.lower()).replace(".xom",'')
                        if ca_present_flag == 1:
                            inp_keyword = (inp_keyword.lower()).replace(".ca",'')
                        if om_present_flag == 1:
                            inp_keyword = (inp_keyword.lower()).replace(".om",'')
                        if "carte" in row_split:
                            carte_flag = 1
                            inp_keyword = (inp_keyword.lower()).replace("carte",'')

                        row["QUERY"] = inp_keyword
                        for drg in drug_names.values():
                            row["QUERY"] = ((row["QUERY"]).lower()).replace(drg,'')

                        if flag == 1:
                            resp = self.cloud_translation(row["QUERY"])
                            if resp.status_code == 200:
                                    response = [
                                        json.loads(resp.content)["data"]["translations"][0][
                                            "translatedText"
                                        ],
                                        json.loads(resp.content)["data"]["translations"][0][
                                            "detectedSourceLanguage"
                                        ],
                                    ]
                                    self.translation_calls = self.translation_calls + 1
                                    new_lang = (
                                        self.country_codes[
                                            self.country_codes["LanguageCode"] == response[1]
                                        ]["Language"]
                                        .head(1)
                                        .values
                                    )
                                    if new_lang:
                                        language = new_lang[0]
                                    else:
                                        language = "Unassigned"
                                    if response[1] == 'en':
                                        for idx,drg in drug_names.items():
                                            row["QUERY"] = row["QUERY"][:idx] + drg + row["QUERY"][idx:]
                                        if abbr_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,abbr in abbr_dict.items():
                                                row_split_words.insert(idx,abbr)
                                            row["QUERY"] = " ".join(row_split_words)
                                        
                                        if com_word_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,com in com_word_dict.items():
                                                row_split_words.insert(idx,com)
                                            row["QUERY"] = " ".join(row_split_words)
                                        if xom_word_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,xom in xom_word_dict.items():
                                                row_split_words.insert(idx,xom)
                                            row["QUERY"] = " ".join(row_split_words)
                                        
                                        if com_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                        if com_w_comma_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                        if xom_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                        if ca_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                        if om_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:om_index] + ".om" + row["QUERY"][om_index:]
                                        if carte_flag == 1:
                                            row["QUERY"] = row["QUERY"] + " card"
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "fr",
                                                    "QUERY_LANGUAGE": "French",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                        else:
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "en",
                                                    "QUERY_LANGUAGE": "English",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                    if row["QUERY"] == response[0] and row["IS_BRANDED"] == True and language != 'en'\
                                        and ((ord(row["QUERY"][0]) in range(0, 123)) 
                                                or (ord(row["QUERY"][-1]) in range(0, 123))):
                                        for idx,drg in drug_names.items():
                                            row["QUERY"] = row["QUERY"][:idx] + drg + row["QUERY"][idx:]
                                        if abbr_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,abbr in abbr_dict.items():
                                                row_split_words.insert(idx,abbr)
                                            row["QUERY"] = " ".join(row_split_words)
                                        
                                        if com_word_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,com in com_word_dict.items():
                                                row_split_words.insert(idx,com)
                                            row["QUERY"] = " ".join(row_split_words)
                                        if xom_word_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,xom in xom_word_dict.items():
                                                row_split_words.insert(idx,xom)
                                            row["QUERY"] = " ".join(row_split_words)
                                        
                                        if com_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                        if com_w_comma_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                        if xom_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                        if ca_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                        if om_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:om_index] + ".om" + row["QUERY"][om_index:]
                                        if carte_flag == 1:
                                            row["QUERY"] = row["QUERY"] + " card"
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "fr",
                                                    "QUERY_LANGUAGE": "French",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                        else:
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "en",
                                                    "QUERY_LANGUAGE": "English",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                    else:
                                            resp_w_drug = response[0]
                                            for drg in drug_names.values():
                                                resp_w_drug = resp_w_drug + " " + drg
                                            if com_present_flag == 1:
                                                if word_before_com in resp_w_drug:
                                                    resp_w_drug = resp_w_drug.replace(word_before_com,(word_before_com+".com"))
                                                else:
                                                    resp_w_drug = resp_w_drug + ".com"
                                            if com_w_comma_present_flag == 1:
                                                if word_before_com_w_comma in resp_w_drug:
                                                    resp_w_drug = resp_w_drug.replace(word_before_com_w_comma,(word_before_com_w_comma+",com"))
                                                else:
                                                    resp_w_drug = resp_w_drug + ",com"
                                            if xom_present_flag == 1:
                                                if word_before_xom in resp_w_drug:
                                                    resp_w_drug = resp_w_drug.replace(word_before_xom,(word_before_xom+".xom"))
                                                else:
                                                    resp_w_drug = resp_w_drug + ".xom"
                                            if ca_present_flag == 1:
                                                if word_before_ca in resp_w_drug:
                                                    resp_w_drug = resp_w_drug.replace(word_before_ca,(word_before_ca+".ca"))
                                                else:
                                                    resp_w_drug = resp_w_drug + ".ca"
                                            if om_present_flag == 1:
                                                if word_before_om in resp_w_drug:
                                                    resp_w_drug = resp_w_drug.replace(word_before_om,(word_before_om+".om"))
                                                else:
                                                    resp_w_drug = resp_w_drug + ".om"
                                            if abbr_present_flag == 1:
                                                for idx,abbr in abbr_dict.items():
                                                    resp_w_drug = resp_w_drug + " " + abbr
                                            if com_word_present_flag == 1:
                                                for idx,com in com_word_dict.items():
                                                    resp_w_drug = resp_w_drug + " " + com
                                            if xom_word_present_flag == 1:
                                                for idx,xom in xom_word_dict.items():
                                                    resp_w_drug = resp_w_drug + " " + xom

                                            if carte_flag == 1:
                                                resp_w_drug = resp_w_drug + " card"
                                                return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": resp_w_drug,
                                                    "QUERY_LANGUAGE_CODE": "fr",
                                                    "QUERY_LANGUAGE": "French",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                                )
                                            else:
                                                return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": resp_w_drug,
                                                    "QUERY_LANGUAGE_CODE": response[1],
                                                    "QUERY_LANGUAGE": language,
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                                )
                            else:
                                self.failed_translation_calls = (
                                        self.failed_translation_calls + 1
                                    )
                                for idx,drg in drug_names.items():
                                    row["QUERY"] = row["QUERY"][:idx] + drg + row["QUERY"][idx:]
                                if abbr_present_flag == 1:
                                    row_split_words = (row["QUERY"]).split(" ")
                                    for idx,abbr in abbr_dict.items():
                                        row_split_words.insert(idx,abbr)
                                    row["QUERY"] = " ".join(row_split_words)
                                
                                if com_word_present_flag == 1:
                                    row_split_words = (row["QUERY"]).split(" ")
                                    for idx,com in com_word_dict.items():
                                        row_split_words.insert(idx,com)
                                    row["QUERY"] = " ".join(row_split_words)
                                if xom_word_present_flag == 1:
                                    row_split_words = (row["QUERY"]).split(" ")
                                    for idx,xom in xom_word_dict.items():
                                        row_split_words.insert(idx,xom)
                                    row["QUERY"] = " ".join(row_split_words)
                                
                                if com_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                if com_w_comma_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                if xom_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                if ca_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                if carte_flag == 1:
                                    row["QUERY"] = row["QUERY"] + " card"
                                    return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "fr",
                                                    "QUERY_LANGUAGE": "French",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                else:
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "en",
                                                    "QUERY_LANGUAGE": "English",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )

                        if flag == 0:             #No drug present
                                resp = self.cloud_translation(row["QUERY"])
                                if resp.status_code == 200:
                                    response = [
                                        json.loads(resp.content)["data"]["translations"][0][
                                            "translatedText"
                                        ],
                                        json.loads(resp.content)["data"]["translations"][0][
                                            "detectedSourceLanguage"
                                        ],
                                    ]
                                    self.translation_calls = self.translation_calls + 1
                                    new_lang = (
                                        self.country_codes[
                                            self.country_codes["LanguageCode"] == response[1]
                                        ]["Language"]
                                        .head(1)
                                        .values
                                    )
                                    if new_lang:
                                        language = new_lang[0]
                                    else:
                                        language = "Unassigned"
                                    if response[1] == 'en':
                                        if abbr_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,abbr in abbr_dict.items():
                                                row_split_words.insert(idx,abbr)
                                            row["QUERY"] = " ".join(row_split_words)
                                        
                                        if com_word_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,com in com_word_dict.items():
                                                row_split_words.insert(idx,com)
                                            row["QUERY"] = " ".join(row_split_words)
                                        if xom_word_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,xom in xom_word_dict.items():
                                                row_split_words.insert(idx,xom)
                                            row["QUERY"] = " ".join(row_split_words)

                                        if com_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                        if com_w_comma_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                        if xom_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                        if ca_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                        if om_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:om_index] + ".om" + row["QUERY"][om_index:]
                                        if carte_flag == 1:
                                            row["QUERY"] = row["QUERY"] + " card"
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "fr",
                                                    "QUERY_LANGUAGE": "French",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                        else:
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "en",
                                                    "QUERY_LANGUAGE": "English",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                    if row["QUERY"] == response[0] and row["IS_BRANDED"] == True and language != 'en'\
                                        and ((ord(row["QUERY"][0]) in range(0, 123)) 
                                                or (ord(row["QUERY"][-1]) in range(0, 123))):
                                        if abbr_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,abbr in abbr_dict.items():
                                                row_split_words.insert(idx,abbr)
                                            row["QUERY"] = " ".join(row_split_words)
                                        if com_word_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,com in com_word_dict.items():
                                                row_split_words.insert(idx,com)
                                            row["QUERY"] = " ".join(row_split_words)
                                        if xom_word_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,xom in xom_word_dict.items():
                                                row_split_words.insert(idx,xom)
                                            row["QUERY"] = " ".join(row_split_words)
                                        
                                        if com_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                        if com_w_comma_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                        if xom_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                        if ca_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                        if om_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:om_index] + ".om" + row["QUERY"][om_index:]
                                        if carte_flag == 1:
                                            row["QUERY"] = row["QUERY"] + " card"
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "fr",
                                                    "QUERY_LANGUAGE": "French",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                        else:
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "en",
                                                    "QUERY_LANGUAGE": "English",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                    else:
                                            resp_w_drug = response[0]
                                            if com_present_flag == 1:
                                                if word_before_com in resp_w_drug:
                                                    resp_w_drug = resp_w_drug.replace(word_before_com,(word_before_com+".com"))
                                                else:
                                                    resp_w_drug = resp_w_drug + ".com"
                                            if com_w_comma_present_flag == 1:
                                                if word_before_com_w_comma in resp_w_drug:
                                                    resp_w_drug = resp_w_drug.replace(word_before_com_w_comma,(word_before_com_w_comma+",com"))
                                                else:
                                                    resp_w_drug = resp_w_drug + ",com"
                                            if xom_present_flag == 1:
                                                if word_before_xom in resp_w_drug:
                                                    resp_w_drug = resp_w_drug.replace(word_before_xom,(word_before_xom+".xom"))
                                                else:
                                                    resp_w_drug = resp_w_drug + ".xom"
                                            if ca_present_flag == 1:
                                                if word_before_ca in resp_w_drug:
                                                    resp_w_drug = resp_w_drug.replace(word_before_ca,(word_before_ca+".ca"))
                                                else:
                                                    resp_w_drug = resp_w_drug + ".ca"
                                            if om_present_flag == 1:
                                                if word_before_om in resp_w_drug:
                                                    resp_w_drug = resp_w_drug.replace(word_before_om,(word_before_om+".om"))
                                                else:
                                                    resp_w_drug = resp_w_drug + ".om"
                                            if abbr_present_flag == 1:
                                                for idx,abbr in abbr_dict.items():
                                                    resp_w_drug = resp_w_drug + " " + abbr
                                            
                                            if com_word_present_flag == 1:
                                                for idx,com in com_word_dict.items():
                                                    resp_w_drug = resp_w_drug + " " + com
                                            if xom_word_present_flag == 1:
                                                for idx,xom in xom_word_dict.items():
                                                    resp_w_drug = resp_w_drug + " " + xom
                                            if carte_flag == 1:
                                                resp_w_drug = resp_w_drug + " card"
                                                return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": resp_w_drug,
                                                    "QUERY_LANGUAGE_CODE": "fr",
                                                    "QUERY_LANGUAGE": "French",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                                )
                                            else:
                                                return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": resp_w_drug,
                                                    "QUERY_LANGUAGE_CODE": response[1],
                                                    "QUERY_LANGUAGE": language,
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                                )
                                else:
                                    self.failed_translation_calls = (
                                        self.failed_translation_calls + 1
                                    )
                                    if abbr_present_flag == 1:
                                        row_split_words = (row["QUERY"]).split(" ")
                                        for idx,abbr in abbr_dict.items():
                                            row_split_words.insert(idx,abbr)
                                        row["QUERY"] = " ".join(row_split_words)
                                    if com_word_present_flag == 1:
                                        row_split_words = (row["QUERY"]).split(" ")
                                        for idx,com in com_word_dict.items():
                                            row_split_words.insert(idx,com)
                                        row["QUERY"] = " ".join(row_split_words)
                                    if xom_word_present_flag == 1:
                                        row_split_words = (row["QUERY"]).split(" ")
                                        for idx,xom in xom_word_dict.items():
                                            row_split_words.insert(idx,xom)
                                        row["QUERY"] = " ".join(row_split_words)

                                    if com_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                    if com_w_comma_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                    if xom_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                    if ca_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                    if om_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:om_index] + ".om" + row["QUERY"][om_index:]
                                    if carte_flag == 1:
                                        row["QUERY"] = row["QUERY"] + " card"
                                        return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "fr",
                                                    "QUERY_LANGUAGE": "French",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                    else:
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "en",
                                                    "QUERY_LANGUAGE": "English",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )  
                    #############################################
#  this is the part where ARYK customer is taken care of
                    elif self.CUSTOMER == 'ARYK':
                        flag = 0
                        aryk_list = self.get_aryk_list(self.CUSTOMER)
                        aryk_names = dict()
                        for word in aryk_list:
                            if len((row["QUERY"]).split(" ")) == 1 and (row["QUERY"]).lower() in aryk_list:
                                if com_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                if com_w_comma_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                if xom_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                if ca_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                if abbr_present_flag == 1:
                                    for idx,abbr in abbr_dict.items():
                                            row["QUERY"] = row["QUERY"][:idx] + abbr + row["QUERY"][idx:]
                                if carte_flag == 1:
                                    row["QUERY"] = row["QUERY"] + " card"

                                if "&quot;" in row["QUERY"]:
                                    row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                if "&#39;" in row["QUERY"]:
                                    row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                return pd.Series(
                                            {
                                                "TRANSLATED_QUERY": row["QUERY"],
                                                "QUERY_LANGUAGE_CODE": "en",
                                                "QUERY_LANGUAGE": "English",
                                                "GCP_TRANSLATED_QUERY": row["QUERY"],
                                                "GCP_QUERY_LANGUAGE_CODE": row["QUERY_LANGUAGE_CODE"],
                                                "GCP_QUERY_LANGUAGE": row["QUERY_LANGUAGE"],
                                            }
                                        )
                            if word in (row["QUERY"]).lower():
                                flag = 1
                                index = ((inp_keyword).lower()).index(word)
                                aryk_names[index] = word
                                row["QUERY"] = ((row["QUERY"]).lower()).replace(word,'')

                        aryk_names = collections.OrderedDict(sorted(aryk_names.items()))
                        if abbr_present_flag == 1:
                            inp_keyword_split = inp_keyword.split(" ")
                            for abbr in abbr_dict.values():
                                for i in range(len(inp_keyword_split)):
                                    if inp_keyword_split[i] == abbr:
                                        inp_keyword_split[i] = ''
                                # inp_keyword = (inp_keyword.lower()).replace(abbr,'')
                            inp_keyword = " ".join(inp_keyword_split)
                        if com_word_present_flag == 1:
                            for com in com_word_dict.values():
                                inp_keyword = (inp_keyword.lower()).replace(com,'')
                        if xom_word_present_flag == 1:
                            for xom in xom_word_dict.values():
                                inp_keyword = (inp_keyword.lower()).replace(xom,'')

                        if com_present_flag == 1:
                            inp_keyword = (inp_keyword.lower()).replace(".com",'')
                        if com_w_comma_present_flag == 1:
                            inp_keyword = (inp_keyword.lower()).replace(",com",'')
                        if xom_present_flag == 1:
                            inp_keyword = (inp_keyword.lower()).replace(".xom",'')
                        if ca_present_flag == 1:
                            inp_keyword = (inp_keyword.lower()).replace(".ca",'')
                        if om_present_flag == 1:
                            inp_keyword = (inp_keyword.lower()).replace(".om",'')
                        if "carte" in row_split:
                            carte_flag = 1
                            inp_keyword = (inp_keyword.lower()).replace("carte",'')

                        row["QUERY"] = inp_keyword
                        for word in aryk_names.values():
                            row["QUERY"] = ((row["QUERY"]).lower()).replace(word,'')
                            
                        if flag == 1:
                            resp = self.cloud_translation(row["QUERY"])
                            if resp.status_code == 200:
                                    response = [
                                        json.loads(resp.content)["data"]["translations"][0][
                                            "translatedText"
                                        ],
                                        json.loads(resp.content)["data"]["translations"][0][
                                            "detectedSourceLanguage"
                                        ],
                                    ]
                                    self.translation_calls = self.translation_calls + 1
                                    new_lang = (
                                        self.country_codes[
                                            self.country_codes["LanguageCode"] == response[1]
                                        ]["Language"]
                                        .head(1)
                                        .values
                                    )
                                    if new_lang:
                                        language = new_lang[0]
                                    else:
                                        language = "Unassigned"
                                    if response[1] == 'en':
                                        for idx,word in aryk_names.items():
                                            row["QUERY"] = row["QUERY"][:idx] + word + row["QUERY"][idx:]
                                        if abbr_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,abbr in abbr_dict.items():
                                                row_split_words.insert(idx,abbr)
                                            row["QUERY"] = " ".join(row_split_words)
                                        
                                        if com_word_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,com in com_word_dict.items():
                                                row_split_words.insert(idx,com)
                                            row["QUERY"] = " ".join(row_split_words)
                                        if xom_word_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,xom in xom_word_dict.items():
                                                row_split_words.insert(idx,xom)
                                            row["QUERY"] = " ".join(row_split_words)
                                        
                                        if com_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                        if com_w_comma_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                        if xom_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                        if ca_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                        if om_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:om_index] + ".om" + row["QUERY"][om_index:]
                                        if "&quot;" in row["QUERY"]:
                                            row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                        if "&#39;" in row["QUERY"]:
                                            row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                        if carte_flag == 1:
                                            row["QUERY"] = row["QUERY"] + " card"
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "fr",
                                                    "QUERY_LANGUAGE": "French",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                        else:
                                            if "&quot;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                            if "&#39;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "en",
                                                    "QUERY_LANGUAGE": "English",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                    if row["QUERY"] == response[0] and row["IS_BRANDED"] == True and language != 'en'\
                                        and ((ord(row["QUERY"][0]) in range(0, 123)) 
                                                or (ord(row["QUERY"][-1]) in range(0, 123))):
                                        for idx,word in aryk_names.items():
                                            row["QUERY"] = row["QUERY"][:idx] + word + row["QUERY"][idx:]
                                        if abbr_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,abbr in abbr_dict.items():
                                                row_split_words.insert(idx,abbr)
                                            row["QUERY"] = " ".join(row_split_words)
                                        
                                        if com_word_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,com in com_word_dict.items():
                                                row_split_words.insert(idx,com)
                                            row["QUERY"] = " ".join(row_split_words)
                                        if xom_word_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,xom in xom_word_dict.items():
                                                row_split_words.insert(idx,xom)
                                            row["QUERY"] = " ".join(row_split_words)
                                        
                                        if com_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                        if com_w_comma_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                        if xom_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                        if ca_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                        if om_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:om_index] + ".om" + row["QUERY"][om_index:]
                                        if "&quot;" in row["QUERY"]:
                                            row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                        if "&#39;" in row["QUERY"]:
                                            row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                        if carte_flag == 1:
                                            row["QUERY"] = row["QUERY"] + " card"
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "fr",
                                                    "QUERY_LANGUAGE": "French",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                        else:
                                            if "&quot;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                            if "&#39;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "en",
                                                    "QUERY_LANGUAGE": "English",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                    else:
                                            resp_w_word = response[0]
                                            for word in aryk_names.values():
                                                resp_w_word = resp_w_word + " " + word
                                            if com_present_flag == 1:
                                                if word_before_com in resp_w_word:
                                                    resp_w_word = resp_w_word.replace(word_before_com,(word_before_com+".com"))
                                                else:
                                                    resp_w_word = resp_w_word + ".com"
                                            if com_w_comma_present_flag == 1:
                                                if word_before_com_w_comma in resp_w_word:
                                                    resp_w_word = resp_w_word.replace(word_before_com_w_comma,(word_before_com_w_comma+",com"))
                                                else:
                                                    resp_w_word = resp_w_word + ",com"
                                            if xom_present_flag == 1:
                                                if word_before_xom in resp_w_word:
                                                    resp_w_word = resp_w_word.replace(word_before_xom,(word_before_xom+".xom"))
                                                else:
                                                    resp_w_word = resp_w_word + ".xom"
                                            if ca_present_flag == 1:
                                                if word_before_ca in resp_w_word:
                                                    resp_w_word = resp_w_word.replace(word_before_ca,(word_before_ca+".ca"))
                                                else:
                                                    resp_w_word = resp_w_word + ".ca"
                                            if om_present_flag == 1:
                                                if word_before_om in resp_w_word:
                                                    resp_w_word = resp_w_word.replace(word_before_om,(word_before_om+".om"))
                                                else:
                                                    resp_w_word = resp_w_word + ".om"
                                            if abbr_present_flag == 1:
                                                for idx,abbr in abbr_dict.items():
                                                    resp_w_word = resp_w_word + " " + abbr
                                            if com_word_present_flag == 1:
                                                for idx,com in com_word_dict.items():
                                                    resp_w_word = resp_w_word + " " + com
                                            if xom_word_present_flag == 1:
                                                for idx,xom in xom_word_dict.items():
                                                    resp_w_word = resp_w_word + " " + xom
                                            if "&quot;" in resp_w_word:
                                                resp_w_word = (resp_w_word).replace("&quot;",'"')
                                            if "&#39;" in resp_w_word:
                                                resp_w_word = (resp_w_word).replace("&#39;","'")
                                            if carte_flag == 1:
                                                resp_w_word = resp_w_word + " card"
                                                return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": resp_w_word,
                                                    "QUERY_LANGUAGE_CODE": "fr",
                                                    "QUERY_LANGUAGE": "French",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                                )
                                            else:
                                                if "&quot;" in resp_w_word:
                                                    resp_w_word = (resp_w_word).replace("&quot;",'"')
                                                if "&#39;" in resp_w_word:
                                                    resp_w_word = (resp_w_word).replace("&#39;","'")
                                                return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": resp_w_word,
                                                    "QUERY_LANGUAGE_CODE": response[1],
                                                    "QUERY_LANGUAGE": language,
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                                )
                            else:
                                self.failed_translation_calls = (
                                        self.failed_translation_calls + 1
                                    )
                                for idx,word in aryk_names.items():
                                    row["QUERY"] = row["QUERY"][:idx] + word + row["QUERY"][idx:]
                                if abbr_present_flag == 1:
                                    row_split_words = (row["QUERY"]).split(" ")
                                    for idx,abbr in abbr_dict.items():
                                        row_split_words.insert(idx,abbr)
                                    row["QUERY"] = " ".join(row_split_words)
                                
                                if com_word_present_flag == 1:
                                    row_split_words = (row["QUERY"]).split(" ")
                                    for idx,com in com_word_dict.items():
                                        row_split_words.insert(idx,com)
                                    row["QUERY"] = " ".join(row_split_words)
                                if xom_word_present_flag == 1:
                                    row_split_words = (row["QUERY"]).split(" ")
                                    for idx,xom in xom_word_dict.items():
                                        row_split_words.insert(idx,xom)
                                    row["QUERY"] = " ".join(row_split_words)
                                
                                if com_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                if com_w_comma_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                if xom_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                if ca_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                if "&quot;" in row["QUERY"]:
                                    row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                if "&#39;" in row["QUERY"]:
                                    row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                if carte_flag == 1:
                                    row["QUERY"] = row["QUERY"] + " card"
                                    return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "fr",
                                                    "QUERY_LANGUAGE": "French",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                else:
                                            if "&quot;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                            if "&#39;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "en",
                                                    "QUERY_LANGUAGE": "English",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )

                        if flag == 0:             #No ARYK imp word present
                                resp = self.cloud_translation(row["QUERY"])
                                if resp.status_code == 200:
                                    response = [
                                        json.loads(resp.content)["data"]["translations"][0][
                                            "translatedText"
                                        ],
                                        json.loads(resp.content)["data"]["translations"][0][
                                            "detectedSourceLanguage"
                                        ],
                                    ]
                                    self.translation_calls = self.translation_calls + 1
                                    new_lang = (
                                        self.country_codes[
                                            self.country_codes["LanguageCode"] == response[1]
                                        ]["Language"]
                                        .head(1)
                                        .values
                                    )
                                    if new_lang:
                                        language = new_lang[0]
                                    else:
                                        language = "Unassigned"
                                    if response[1] == 'en':
                                        if abbr_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,abbr in abbr_dict.items():
                                                row_split_words.insert(idx,abbr)
                                            row["QUERY"] = " ".join(row_split_words)
                                        
                                        if com_word_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,com in com_word_dict.items():
                                                row_split_words.insert(idx,com)
                                            row["QUERY"] = " ".join(row_split_words)
                                        if xom_word_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,xom in xom_word_dict.items():
                                                row_split_words.insert(idx,xom)
                                            row["QUERY"] = " ".join(row_split_words)

                                        if com_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                        if com_w_comma_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                        if xom_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                        if ca_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                        if om_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:om_index] + ".om" + row["QUERY"][om_index:]
                                        if "&quot;" in row["QUERY"]:
                                            row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                        if "&#39;" in row["QUERY"]:
                                            row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                        if carte_flag == 1:
                                            row["QUERY"] = row["QUERY"] + " card"
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "fr",
                                                    "QUERY_LANGUAGE": "French",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                        else:
                                            if "&quot;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                            if "&#39;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "en",
                                                    "QUERY_LANGUAGE": "English",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                    if row["QUERY"] == response[0] and row["IS_BRANDED"] == True and language != 'en'\
                                        and ((ord(row["QUERY"][0]) in range(0, 123)) 
                                                or (ord(row["QUERY"][-1]) in range(0, 123))):
                                        if abbr_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,abbr in abbr_dict.items():
                                                row_split_words.insert(idx,abbr)
                                            row["QUERY"] = " ".join(row_split_words)
                                        if com_word_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,com in com_word_dict.items():
                                                row_split_words.insert(idx,com)
                                            row["QUERY"] = " ".join(row_split_words)
                                        if xom_word_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,xom in xom_word_dict.items():
                                                row_split_words.insert(idx,xom)
                                            row["QUERY"] = " ".join(row_split_words)
                                        
                                        if com_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                        if com_w_comma_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                        if xom_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                        if ca_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                        if om_present_flag == 1:
                                            row["QUERY"] = row["QUERY"][:om_index] + ".om" + row["QUERY"][om_index:]
                                        if "&quot;" in row["QUERY"]:
                                            row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                        if "&#39;" in row["QUERY"]:
                                            row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                        if carte_flag == 1:
                                            row["QUERY"] = row["QUERY"] + " card"
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "fr",
                                                    "QUERY_LANGUAGE": "French",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                        else:
                                            if "&quot;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                            if "&#39;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "en",
                                                    "QUERY_LANGUAGE": "English",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                    else:
                                            resp_w_word = response[0]
                                            if com_present_flag == 1:
                                                if word_before_com in resp_w_word:
                                                    resp_w_word = resp_w_word.replace(word_before_com,(word_before_com+".com"))
                                                else:
                                                    resp_w_word = resp_w_word + ".com"
                                            if com_w_comma_present_flag == 1:
                                                if word_before_com_w_comma in resp_w_word:
                                                    resp_w_word = resp_w_word.replace(word_before_com_w_comma,(word_before_com_w_comma+",com"))
                                                else:
                                                    resp_w_word = resp_w_word + ",com"
                                            if xom_present_flag == 1:
                                                if word_before_xom in resp_w_word:
                                                    resp_w_word = resp_w_word.replace(word_before_xom,(word_before_xom+".xom"))
                                                else:
                                                    resp_w_word = resp_w_word + ".xom"
                                            if ca_present_flag == 1:
                                                if word_before_ca in resp_w_word:
                                                    resp_w_word = resp_w_word.replace(word_before_ca,(word_before_ca+".ca"))
                                                else:
                                                    resp_w_word = resp_w_word + ".ca"
                                            if om_present_flag == 1:
                                                if word_before_om in resp_w_word:
                                                    resp_w_word = resp_w_word.replace(word_before_om,(word_before_om+".om"))
                                                else:
                                                    resp_w_word = resp_w_word + ".om"
                                            if abbr_present_flag == 1:
                                                for idx,abbr in abbr_dict.items():
                                                    resp_w_word = resp_w_word + " " + abbr
                                            
                                            if com_word_present_flag == 1:
                                                for idx,com in com_word_dict.items():
                                                    resp_w_word = resp_w_word + " " + com
                                            if xom_word_present_flag == 1:
                                                for idx,xom in xom_word_dict.items():
                                                    resp_w_word = resp_w_word + " " + xom
                                            if "&quot;" in resp_w_word:
                                                resp_w_word = (resp_w_word).replace("&quot;",'"')
                                            if "&#39;" in resp_w_word:
                                                resp_w_word = (resp_w_word).replace("&#39;","'")
                                            if carte_flag == 1:
                                                resp_w_word = resp_w_word + " card"
                                                return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": resp_w_word,
                                                    "QUERY_LANGUAGE_CODE": "fr",
                                                    "QUERY_LANGUAGE": "French",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                                )
                                            else:
                                                if "&quot;" in resp_w_word:
                                                    resp_w_word = (resp_w_word).replace("&quot;",'"')
                                                if "&#39;" in resp_w_word:
                                                    resp_w_word = (resp_w_word).replace("&#39;","'")
                                                return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": resp_w_word,
                                                    "QUERY_LANGUAGE_CODE": response[1],
                                                    "QUERY_LANGUAGE": language,
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                                )
                                else:
                                    self.failed_translation_calls = (
                                        self.failed_translation_calls + 1
                                    )
                                    if abbr_present_flag == 1:
                                        row_split_words = (row["QUERY"]).split(" ")
                                        for idx,abbr in abbr_dict.items():
                                            row_split_words.insert(idx,abbr)
                                        row["QUERY"] = " ".join(row_split_words)
                                    if com_word_present_flag == 1:
                                        row_split_words = (row["QUERY"]).split(" ")
                                        for idx,com in com_word_dict.items():
                                            row_split_words.insert(idx,com)
                                        row["QUERY"] = " ".join(row_split_words)
                                    if xom_word_present_flag == 1:
                                        row_split_words = (row["QUERY"]).split(" ")
                                        for idx,xom in xom_word_dict.items():
                                            row_split_words.insert(idx,xom)
                                        row["QUERY"] = " ".join(row_split_words)

                                    if com_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                    if com_w_comma_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                    if xom_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                    if ca_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                    if om_present_flag == 1:
                                        row["QUERY"] = row["QUERY"][:om_index] + ".om" + row["QUERY"][om_index:]
                                    if "&quot;" in row["QUERY"]:
                                        row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                    if "&#39;" in row["QUERY"]:
                                        row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                    if carte_flag == 1:
                                        row["QUERY"] = row["QUERY"] + " card"
                                        return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "fr",
                                                    "QUERY_LANGUAGE": "French",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                                    else:
                                            if "&quot;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                            if "&#39;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "en",
                                                    "QUERY_LANGUAGE": "English",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                    
#  this is for others customer
                    else:
                        resp = self.cloud_translation(row["QUERY"])
                        if resp.status_code == 200:
                            response = [
                                json.loads(resp.content)["data"]["translations"][0][
                                    "translatedText"
                                ],
                                json.loads(resp.content)["data"]["translations"][0][
                                    "detectedSourceLanguage"
                                ],
                            ]
                            self.translation_calls = self.translation_calls + 1
                            new_lang = (
                                self.country_codes[
                                    self.country_codes["LanguageCode"] == response[1]
                                ]["Language"]
                                .head(1)
                                .values
                            )
                            if new_lang:
                                language = new_lang[0]
                            else:
                                language = "Unassigned"

                            if row["QUERY"] == response[0] and row["IS_BRANDED"] == True and language != 'en'\
                                and ((ord(row["QUERY"][0]) in range(0, 123)) 
                                        or (ord(row["QUERY"][-1]) in range(0, 123))):
                                if abbr_present_flag == 1:
                                    row_split_words = (row["QUERY"]).split(" ")
                                    for idx,abbr in abbr_dict.items():
                                        row_split_words.insert(idx,abbr)
                                    row["QUERY"] = " ".join(row_split_words)
                                if com_word_present_flag == 1:
                                            row_split_words = (row["QUERY"]).split(" ")
                                            for idx,com in com_word_dict.items():
                                                row_split_words.insert(idx,com)
                                            row["QUERY"] = " ".join(row_split_words)
                                if xom_word_present_flag == 1:
                                    row_split_words = (row["QUERY"]).split(" ")
                                    for idx,xom in xom_word_dict.items():
                                        row_split_words.insert(idx,xom)
                                    row["QUERY"] = " ".join(row_split_words)

                                if com_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                                if com_w_comma_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                                if xom_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                                if ca_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                                if om_present_flag == 1:
                                    row["QUERY"] = row["QUERY"][:om_index] + ".om" + row["QUERY"][om_index:]
                                if "&quot;" in row["QUERY"]:
                                    row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                if "&#39;" in row["QUERY"]:
                                    row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                if carte_flag == 1:
                                    row["QUERY"] = row["QUERY"] + " card"
                                    return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "fr",
                                                    "QUERY_LANGUAGE": "French",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )


                                else:
                                            if "&quot;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                            if "&#39;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "en",
                                                    "QUERY_LANGUAGE": "English",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )

                            else:
                                    resp_w_drug = response[0]
                                    if com_present_flag == 1:
                                        if word_before_com in resp_w_drug:
                                            resp_w_drug = resp_w_drug.replace(word_before_com,(word_before_com+".com"))
                                        else:
                                            resp_w_drug = resp_w_drug + ".com"
                                    if com_w_comma_present_flag == 1:
                                        if word_before_com_w_comma in resp_w_drug:
                                            resp_w_drug = resp_w_drug.replace(word_before_com_w_comma,(word_before_com_w_comma+",com"))
                                        else:
                                            resp_w_drug = resp_w_drug + ",com"
                                    if xom_present_flag == 1:
                                        if word_before_xom in resp_w_drug:
                                            resp_w_drug = resp_w_drug.replace(word_before_xom,(word_before_xom+".xom"))
                                        else:
                                            resp_w_drug = resp_w_drug + ".xom"
                                    if ca_present_flag == 1:
                                        if word_before_ca in resp_w_drug:
                                            resp_w_drug = resp_w_drug.replace(word_before_ca,(word_before_ca+".ca"))
                                        else:
                                            resp_w_drug = resp_w_drug + ".ca"
                                    if om_present_flag == 1:
                                        if word_before_om in resp_w_drug:
                                            resp_w_drug = resp_w_drug.replace(word_before_om,(word_before_om+".om"))
                                        else:
                                            resp_w_drug = resp_w_drug + ".om"
                                    if abbr_present_flag == 1:
                                        for idx,abbr in abbr_dict.items():
                                            resp_w_drug = resp_w_drug + " " +abbr
                                    
                                    if com_word_present_flag == 1:
                                        for idx,com in com_word_dict.items():
                                            resp_w_drug = resp_w_drug + " " + com
                                    if xom_word_present_flag == 1:
                                        for idx,xom in xom_word_dict.items():
                                            resp_w_drug = resp_w_drug + " " + xom
                                    if "&quot;" in resp_w_drug:
                                        resp_w_drug = (resp_w_drug).replace("&quot;",'"')
                                    if "&#39;" in resp_w_drug:
                                        resp_w_drug = (resp_w_drug).replace("&#39;","'")
                                    if carte_flag == 1:
                                        resp_w_drug = resp_w_drug + " card"
                                        return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": resp_w_drug,
                                                    "QUERY_LANGUAGE_CODE": "fr",
                                                    "QUERY_LANGUAGE": "French",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                                )
                                    else:
                                                if "&quot;" in resp_w_drug:
                                                    resp_w_drug = (resp_w_drug).replace("&quot;",'"')
                                                if "&#39;" in resp_w_drug:
                                                    resp_w_drug = (resp_w_drug).replace("&#39;","'")
                                                return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": resp_w_drug,
                                                    "QUERY_LANGUAGE_CODE": response[1],
                                                    "QUERY_LANGUAGE": language,
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                                )
                        else:
                            self.failed_translation_calls = (
                                self.failed_translation_calls + 1
                            )
                            if abbr_present_flag == 1:
                                row_split_words = (row["QUERY"]).split(" ")
                                for idx,abbr in abbr_dict.items():
                                    row_split_words.insert(idx,abbr)
                                row["QUERY"] = " ".join(row_split_words)
                            if com_word_present_flag == 1:
                                row_split_words = (row["QUERY"]).split(" ")
                                for idx,com in com_word_dict.items():
                                    row_split_words.insert(idx,com)
                                row["QUERY"] = " ".join(row_split_words)
                            if xom_word_present_flag == 1:
                                row_split_words = (row["QUERY"]).split(" ")
                                for idx,xom in xom_word_dict.items():
                                    row_split_words.insert(idx,xom)
                                row["QUERY"] = " ".join(row_split_words)

                            if com_present_flag == 1:
                                row["QUERY"] = row["QUERY"][:com_index] + ".com" + row["QUERY"][com_index:]
                            if com_w_comma_present_flag == 1:
                                row["QUERY"] = row["QUERY"][:com_w_comma_index] + ",com" + row["QUERY"][com_w_comma_index:]
                            if xom_present_flag == 1:
                                row["QUERY"] = row["QUERY"][:xom_index] + ".xom" + row["QUERY"][xom_index:]
                            if ca_present_flag == 1:
                                row["QUERY"] = row["QUERY"][:ca_index] + ".ca" + row["QUERY"][ca_index:]
                            if om_present_flag == 1:
                                row["QUERY"] = row["QUERY"][:om_index] + ".om" + row["QUERY"][om_index:]
                            if "&quot;" in row["QUERY"]:
                                row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                            if "&#39;" in row["QUERY"]:
                                row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                            if carte_flag == 1:
                                row["QUERY"] = row["QUERY"] + " card"
                                return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "fr",
                                                    "QUERY_LANGUAGE": "French",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                            else:
                                            if "&quot;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                                            if "&#39;" in row["QUERY"]:
                                                row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                                            return pd.Series(
                                                {
                                                    "TRANSLATED_QUERY": row["QUERY"],
                                                    "QUERY_LANGUAGE_CODE": "en",
                                                    "QUERY_LANGUAGE": "English",
                                                    "GCP_TRANSLATED_QUERY": response[0],
                                                    "GCP_QUERY_LANGUAGE_CODE": response[1],
                                                    "GCP_QUERY_LANGUAGE": language,
                                                }
                                            )
                
# this section is for passby translation                
                else:
                    self.passby_translation_calls = self.passby_translation_calls + 1
                    if "&quot;" in row["QUERY"]:
                        row["QUERY"] = (row["QUERY"]).replace("&quot;",'"')
                    if "&#39;" in row["QUERY"]:
                        row["QUERY"] = (row["QUERY"]).replace("&#39;","'")
                    return pd.Series(
                        {
                            "TRANSLATED_QUERY": row["QUERY"],
                            "QUERY_LANGUAGE_CODE": row["QUERY_LANGUAGE_CODE"],
                            "QUERY_LANGUAGE": row["QUERY_LANGUAGE"],
                            "GCP_TRANSLATED_QUERY": None,
                            "GCP_QUERY_LANGUAGE_CODE": None,
                            "GCP_QUERY_LANGUAGE": None,
                        }
                    )
            
# this section is for failed translation       
            except Exception as e:
                self.failed_translation_calls = self.failed_translation_calls + 1
                logging.error("Translation Error %s" % (e))
                self.update_intent_table()
                if (self.translation_calls / self.BATCH_SIZE).is_integer():
                    logging.info(self.translation_calls)
                return pd.Series(
                    {
                        "TRANSLATED_QUERY": None,
                        "QUERY_LANGUAGE_CODE": row["QUERY_LANGUAGE_CODE"],
                        "QUERY_LANGUAGE": row["QUERY_LANGUAGE"],
                        "GCP_TRANSLATED_QUERY": None,
                        "GCP_QUERY_LANGUAGE_CODE": None,
                        "GCP_QUERY_LANGUAGE": None,
                    }
                )

def main():
    lt = LanguageDetectorTranslator()
    lt.load_new_keywords()
    if not lt.keyword_df.empty:
        lt.keyword_df[
            [
                "TRANSLATED_QUERY",
                "QUERY_LANGUAGE_CODE",
                "QUERY_LANGUAGE",
                "GCP_TRANSLATED_QUERY",
                "GCP_QUERY_LANGUAGE_CODE",
                "GCP_QUERY_LANGUAGE",
            ]] = lt.keyword_df.apply(lt.get_translation_independent, axis=1)
        logging.info("Failed Translation Calls %s" % lt.failed_translation_calls)
        logging.info("Skipped Translation Calls %s" % lt.passby_translation_calls)
        logging.info("Translation Calls %s" % lt.translation_calls)
        lt.update_intent_table()
    else:
        logging.info("No New Keywords Found for Translations")
    slack_data = {
        "blocks": [
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": "Translation Job Updates for *{0}*".format( "NVSO"
                            # os.environ["CUSTOMER"] 
                        ),
                    }
                ],
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": "Batch Size:* {0}*".format(1000
                            # os.environ["QUERY_LIMIT"] 
                            ),
                    },
                    {
                        "type": "mrkdwn",
                        "text": "New Keywords for Translations #:* {0}*".format(
                            lt.keyword_df['TRANSLATED_QUERY'].count()
                        ),
                    },
                    {
                        "type": "mrkdwn",
                        "text": "Actual translations:* {0}*".format(
                            lt.translation_calls
                        ),
                    },
                    {
                        "type": "mrkdwn",
                        "text": "Failed translations:* {0}*".format(
                            lt.failed_translation_calls
                        ),
                    },
                    {
                        "type": "mrkdwn",
                        "text": "Skipped translations:* {0}*".format(
                            lt.passby_translation_calls
                        ),
                    },
                    {
                        "type": "mrkdwn",
                        "text": "Temp table reference:* {0}*".format(lt.temp_table),
                    },
                ],
            },
        ]
    }
    try:
        response = requests.post(
            lt.SLACK_WEBHOOK,
            data=json.dumps(slack_data),
            headers={"Content-Type": "application/json"},
        )
    except Exception as e:
        logging.error("Exception in sending slack notifications")
        logging.error(str(e))


if __name__ == "__main__":
    main()



