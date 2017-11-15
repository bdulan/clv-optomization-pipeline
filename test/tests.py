#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_preprocessing
----------------------------------
Tests for `preprocessing` module.
"""

import cx_Oracle
import pandas as pd

def test_expected_result(self,query,expected):

    esi_ids = ['1008901023806376880100',
                '10443720005077463',
                '1008901024900937590115',
                '10443720000476365',
                '10443720007184078',
                '10204049746806890',
                '10443720001410851',
                '10443720009452773',
                '10443720007351845',
                '1008901023812432670102',
                '1008901023805113740100',
                '1008901006900123900107',
                '10443720003806349',
                '10443720007840307',
                '1008901023900817830114',
                '10400513120690001',
                '10443720006205717',
                '1008901018146675370100',
                '10032789471637730']

    connection = cx_Oracle.connect("hr", "welcome", "localhost/orclpdb")

    cursor = connection.cursor()
    cursor.execute("""
                        SELECT 
                        ESI_ID,
                        USAGE01,
                        USAGE02,
                        USAGE03,
                        USAGE04,
                        USAGE05,
                        USAGE06,
                        USAGE07,
                        USAGE08,
                        USAGE09,
                        USAGE10,
                        USAGE11,
                        USAGE12
                        FROM MDRPLSQL.AA_USAGEESID
                        WHERE ESI_ID IN
                        (:esi_ids""",
                        esi_ids = esi_ids
                        )
    for fname, lname in cursor:
	print("Values:", fname, lname)
    


    

    with cx_Oracle.connect(definition) as conn:
        result = pd.read_sql(query, conn)

    tm.assert_frame_equal(result, expected, check_dtype=False, check_like=True)