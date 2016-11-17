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


table_headers = []
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

def AsImgWithBase64(content):
    element = "<img src={0}>".format(content)
    return element

def AddClassName(name):
    table_headers.append(name)

def AddData(predict_data):
    datas.extend(predict_data)

def ProcessToHtml():
    document = BEFORE

    element = AsTableHeader("Image")
    for header in table_headers:
        element += AsTableHeader(header)
    element = AsTableRow(element)
    document += element

    for data in datas:
        element = ""
        img = AsImgWithBase64(data[0])
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
# AddClassName("a")
# AddClassName("b")
# AddData([[ "./Data/sport3/validation/hockey/img_2723.jpg",  0.08581778,  0.30274239]])
# AddData([[ "./Data/sport3/validation/hockey/img_2721.jpg",  0.08581778,  0.30274239]])
# ProcessToHtml()