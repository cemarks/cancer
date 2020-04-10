#!/usr/bin/python3

import re

def check_datatype(value,data_type_str):
    check = False
    if data_type_str in [
        'DATETIME',
        'SAS Date',
        'SAS Time',
        'DATE',
        'ALPHANUMERIC',
        'TIME',
        'HL7EDv3',
        'BOOLEAN',
        'Numeric Alpha DVG',
        'Date Alpha DVG',
        'Derived',
        'UMLUidv1.0',
        'DATE/TIME',
        'HL7STv3',
        'CLOB',
        'HL7CDv3',
        'HL7TSv3',
        'HL7PNv3',
        'HL7TELv3',
        'OBJECT',
        'Alpha DVG'
    ]:
        check = True
    elif data_type_str in ["CHARACTER",'varchar']:
        if re.fullmatch(r"[0-9]*.{0,1}[0-9]*",value):
            return False
        else:
            return True
    elif data_type_str in [
        'Integer',
        'HL7INTv3'
    ]:
        try:
            r = float(value)
            i = int(r)
            if r == i:
                check = True
        except ValueError as e:
            check = False
    elif data_type_str in [
        'NUMBER',
        'HL7REALv3'
    ]:
        try:
            r = float(value)
            check = True
        except ValueError as e:
            check = False
    elif data_type_str == 'binary':
        try:
            r = float(value)
            i = int(r)
            if i in [0,1]:
                check = True
        except ValueError as e:
            check = False
    else:
        check = True
    return check


def check_date_num(month,day):
    check = False
    if (month==2) and (day > 0) and (day < 30):
        check = True
    if (month in [4,6,9,11]) and (day > 0) and (day < 31):
        check = True
    if (month in [1,3,5,7,8,10,12]) and (day > 0) and (day < 32):
        check = True
    return check

def check_hr(hr_int,hr24=False):
    check = False
    if hr24:
        if (hr_int >= 0) and (hr_int <= 24):
            check = True
    else:
        if (hr_int >= 0) and (hr_int <= 12):
            check = True
    return check

def check_min_sec(input_int):
    check = False
    if (input_int >= 0) and (input_int <= 59):
        check = True
    return check

def check_display_format(input_str,display_format):
    check = False
    if display_format in [
        'mm/dd/yy'
    ]: 
        if re.fullmatch(r'[0-9]{1,2}/[0-9]{1,2}/[0-9]{2}',input_str):
            s = input_str.split("/")
            try:
                m = int(s[0])
                d = int(s[1])
            except ValueError as e:
                m = 13
                d = 32
            check = check_date_num(m,d)
    elif display_format == 'DD/MON/YYYY': 
        if re.fullmatch(r'[0-9]{1,2}/[A-Z,a-z]{3}/[0-9]{4}',input_str):
            s = input_str.split("/")
            m_str = s[1].lower()
            if m_str == 'jan':
                m = 1
            elif m_str == 'feb':
                m = 2
            elif m_str == 'mar':
                m = 3
            elif m_str == 'apr':
                m = 4
            elif m_str == 'may':
                m = 5
            elif m_str == 'jun':
                m = 6
            elif m_str == 'jul':
                m = 7
            elif m_str == 'aug':
                m = 8
            elif m_str == 'sep':
                m = 9
            elif m_str == 'oct':
                m = 10
            elif m_str == 'nov':
                m = 11
            elif m-str == 'dec':
                m = 12
            else:
                m = 13
            d = int(s[0])
            check = check_date_num(m,d)
    elif display_format == 'YYYY-MM-DD':
        if re.fullmatch(r'[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}',input_str):
            s = input_str.split("-")
            try:
                m = int(s[1])
                d = int(s[2])
            except ValueError as e:
                m = 13
                d = 32
            check = check_date_num(m,d)
    elif display_format == 'YYYY':
        if re.fullmatch(r'[0-9]{4}',input_str):
            check = True
    elif display_format == 'TIME (HR(24):MN':
        if re.fullmatch(r'[0-9]{1,2}:[0-9]{2}',input_str):
            s = input_str.split(":")
            try:
                h = int(s[0])
                m = int(s[1])
            except ValueError as e:
                h = 25
                m = 61
            if check_hr(h,hr24=True) and check_min_sec(m):
                check = True
    elif display_format == 'YYYYMMDD':
        if re.fullmatch(r'[0-9]{8}',str(input_str)):
            try:
                m = int(str(input_str)[4:6])
                d = int(str(input_str)[6:8])
            except ValueError as e:
                m = 13
                d = 32
            check = check_date_num(m,d)
    elif display_format == '9999.99':
        if re.fullmatch(r'[0-9]{1,4}\.[0-9]{0,2}',str(input_str)) or re.fullmatch(r'[0-9]{1,4}',str(input_str)):
            check = True
    elif display_format in [
        'mm/dd/yyyy',
        'MM/DD/YYYY'
    ]: 
        if re.fullmatch(r'[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}',input_str):
            s = input_str.split("/")
            try:
                m = int(s[0])
                d = int(s[1])
            except ValueError as e:
                m = 13
                d = 32
            check = check_date_num(m,d)
    elif display_format == '9999999':
        if re.fullmatch(r'[0-9]{1,7}',str(input_str)):
            check = True
    elif display_format == '10,3':
        if re.fullmatch(r'[0-9]{1,10}\.[0-9]{0,3}',str(input_str)) or re.fullmatch(r'[0-9]{1,10}',str(input_str)):
            check = True
    elif display_format == '9999.9':
        if re.fullmatch(r'[0-9]{1,4}\.[0-9]{0,1}',str(input_str)) or re.fullmatch(r'[0-9]{1,4}',str(input_str)):
            check = True
    elif display_format == '%':
        if re.fullmatch(r'[0-9]*.{0,1}[0-9]*\%',str(input_str)) or re.fullmatch(r'[0-9]*.{0,1}[0-9]*',str(input_str)):
            check = True
    elif display_format == '999.9':
        if re.fullmatch(r'[0-9]{1,3}\.[0-9]{0,1}',str(input_str)) or re.fullmatch(r'[0-9]{1,3}',str(input_str)):
            check = True
    elif display_format == '99.9':
        if re.fullmatch(r'[0-9]{1,2}\.[0-9]{0,1}',str(input_str)) or re.fullmatch(r'[0-9]{1,2}',str(input_str)):
            check = True
    elif display_format == 'hh:mm:ss':
        if re.fullmatch(r'[0-9]{1,2}:[0-9]{2}:[0-9]{2}',input_str):
            s = input_str.split(":")
            try:
                h = int(s[0])
                m = int(s[1])
                sec = int(s[2])
            except ValueError as e:
                h = 25
                m = 61
                sec = 61
            if check_hr(h,hr24=True) and check_min_sec(m) and check_min_sec(sec):
                check = True
    elif display_format == '999.99':
        if re.fullmatch(r'[0-9]{1,3}\.[0-9]{0,2}',str(input_str)) or re.fullmatch(r'[0-9]{1,3}',str(input_str)):
            check = True
    elif display_format == '9.999':
        if re.fullmatch(r'[0-9]{0,1}\.[0-9]{0,3}',str(input_str)) or re.fullmatch(r'[0-9]{1}',str(input_str)):
            check = True
    elif display_format == '999999.9':
        if re.fullmatch(r'[0-9]{0,6}\.[0-9]{0,1}',str(input_str)) or re.fullmatch(r'[0-9]{1,6}',str(input_str)):
            check = True
    elif display_format in [
        'hh:mm',
        'TIME_HH:MM'
    ]:
        if re.fullmatch(r'[0-9]{1,2}:[0-9]{2}',input_str):
            s = input_str.split(":")
            try:
                h = int(s[0])
                m = int(s[1])
            except ValueError as e:
                h = 25
                m = 61
            if check_hr(h,hr24=True) and check_min_sec(m):
                check = True
    elif display_format == 'hh:mm:ss:rr':
        if re.fullmatch(r'[0-9]{1,2}:[0-9]{2}:[0-9]{2}:[0-9]{1,2}',input_str):
            s = input_str.split(":")
            try:
                h = int(s[0])
                m = int(s[1])
                sec = int(s[2])
            except ValueError as e:
                h = 25
                m = 61
                sec = 61
            if check_hr(h,hr24=True) and check_min_sec(m) and check_min_sec(sec):
                check = True
    elif display_format == 'TIME_MIN':
        check = True
    elif display_format == '99.99':
        if re.fullmatch(r'[0-9]{0,2}\.[0-9]{0,2}',str(input_str)) or re.fullmatch(r'[0-9]{1,2}',str(input_str)):
            check = True
    elif display_format == '9999.999':
        if re.fullmatch(r'[0-9]{0,4}\.[0-9]{0,3}',str(input_str)) or re.fullmatch(r'[0-9]{1,4}',str(input_str)):
            check = True
    elif display_format == 'MMYYYY':
        if re.fullmatch(r'[0-9]{6}',str(input_str)):
            try:
                m = int(str(input_str)[0:2])
            except ValueError as e:
                m = 13
            check = check_date_num(m,1)
    elif display_format == '99999.99':
        if re.fullmatch(r'[0-9]{0,5}\.[0-9]{0,2}',str(input_str)) or re.fullmatch(r'[0-9]{1,5}',str(input_str)):
            check = True
    elif display_format == 'MMDDYYYY':
        if re.fullmatch(r'[0-9]{6}',str(input_str)):
            try:
                m = int(str(input_str)[0:2])
                d = int(str(input_str)[2:4])
            except ValueError as e:
                m = 13
                d = 32
            check = check_date_num(m,d)
    elif display_format == '999-99-9999':
        if re.fullmatch(r'[0-9]{3}-[0-9]{2}-[0-9]{4}',str(input_str)):
            check = True
    elif display_format == 'YYYYMM':
        if re.fullmatch(r'[0-9]{6}',str(input_str)):
            try:
                m = int(str(input_str)[4:6])
            except ValueError as e:
                m = 13
            check = check_date_num(m,1)
    else:
        check = True
    return check

