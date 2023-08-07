SELECT * FROM PERSON
WHERE OUTDATE IS NULL


SELECT D.DEPARTMENT,COUNT(*) COUNT_ FROM PERSON P
JOIN DEPARTMENT D ON D.ID=P.DEPARTMENTID
WHERE P.OUTDATE IS NULL
GROUP BY D.DEPARTMENT

SELECT D.DEPARTMENT,P.GENDER,COUNT(*) COUNT_ FROM PERSON P
JOIN DEPARTMENT D ON D.ID=P.DEPARTMENTID
WHERE P.OUTDATE IS NULL
GROUP BY D.DEPARTMENT,P.GENDER
ORDER BY D.DEPARTMENT


SELECT MIN(SALARY) MIN,MAX(SALARY) MAX,AVG(SALARY) ORTALAMA FROM PERSON P
JOIN DEPARTMENT D ON D.ID=P.DEPARTMENTID
JOIN POSITION PS ON PS.ID=P.POSITIONID
WHERE D.DEPARTMENT='PLANLAMA' AND PS.POSITION LIKE '%�EF%'

SELECT PS.POSITION POZISYON,COUNT(*) KISISAYISI,AVG(P.SALARY) ORTALAMAMAAS FROM PERSON P
JOIN DEPARTMENT D ON D.ID=P.DEPARTMENTID
JOIN POSITION PS ON PS.ID=P.POSITIONID
WHERE OUTDATE IS NULL 
GROUP BY PS.POSITION
ORDER BY ORTALAMAMAAS DESC


SELECT DATEPART(YEAR,P.INDATE) YIL,P.GENDER CINSIYET, COUNT(*) COUNT_ FROM PERSON P
GROUP BY DATEPART(YEAR,P.INDATE),P.GENDER
ORDER BY YIL

SELECT D.DEPARTMENT,AVG(P.SALARY) ORTALAMAMAAS FROM PERSON P
JOIN DEPARTMENT D ON D.ID=P.DEPARTMENTID
GROUP BY D.DEPARTMENT
HAVING AVG(P.SALARY)>5500


SELECT P.NAME_ ISIM,P.SURNAME_ SOYISIM,PS.POSITION POZISYON,PSELF.NAME_ BIRIMYONETICIISIM,
PSELF.SURNAME_ BIRIMYONETICISOYISIM,PS2.POSITION BIRIMYONETICIPOZISYON
FROM PERSON P
JOIN PERSON PSELF ON PSELF.ID=P.PARENTPOSITIONID
JOIN POSITION PS ON PS.ID=P.POSITIONID
JOIN POSITION PS2 ON PS2.ID=PSELF.POSITIONID
ORDER BY P.ID


SELECT T.DEPARTMENT,AVG(MONTH_) ORTALAMASURE FROM(SELECT D.DEPARTMENT,
CASE
	WHEN OUTDATE IS NULL THEN DATEDIFF(MONTH,INDATE,GETDATE())
	ELSE DATEDIFF(MONTH,INDATE,OUTDATE)
END AS MONTH_
FROM PERSON P 
JOIN DEPARTMENT D ON D.ID=P.DEPARTMENTID) T
GROUP BY T.DEPARTMENT
