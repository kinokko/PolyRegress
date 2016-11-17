"""Put validation result into html table"""

BEFORE = """<!DOCTYPE html>
            <html>
            <head>
            <title>Page Title</title>
            </head>
            <body>
            <table>"""
AFTER = """</table>
            </body>
            </html>"""


table_headers = ["basketball", "hockey", "soccer"]
datas = []


def AsTableHeader(content):
    element = "<th>{0}</th>".format(str(content))
    return element

def AsTableData(content):
    element = "<td>{0}</td>".format(str(content))
    return element    

def AsTableRow(content):
    element = "<tr>{0}</tr>".format(str(content))
    return element

def AsImg(content):
    element = "<img src={0}>".format(str(content))
    return element

def AddClassName(name):
    table_headers.append(name)

def AddData(predict_data):
    datas.append(predict_data)

def ProcessToHtml():
    document = BEFORE

    element = AsTableHeader("Image")
    for header in table_headers:
        element += AsTableHeader(header)
    element = AsTableRow(element)
    document += element

    for data in datas:
        element = ""
        img = AsImg(data[0])
        element += AsTableData(img)
        for result in data[1:]:
            element += AsTableData(result)
        element = AsTableRow(element)
        document += element
    
    document += AFTER
    
    f = open("ValidationResult.html", "w")
    f.write(document)
    f.close()


# Test Code
AddData(['./Data/sport3/validation/basketball/img_2164.jpg', 0.34910166, 0.21430533, 0.43659306])
AddData(['./Data/sport3/validation/basketball/img_2156.jpg', 0.40804267, 0.2676276, 0.32432973])
ProcessToHtml()