import csv

def read_csv(file_name):
  #読み込みcsv用意
  csv_file = open(file_name, "r", encoding="ms932", errors="", newline="")
  f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
  #読み込み結果が文字列の配列になってるので、floatの配列に変換する
  flist = []
  for row in f:
    flist.append(list(map(float,row)))
  return flist
