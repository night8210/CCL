import json
import boto3
resource = boto3.resource('dynamodb')
table = resource.Table('myawsdynamo')
def lambda_handler(event, context):
    name = event['firstname'] +' '+ event['lastname']
    point = event['score']
    response = table.put_item(
        Item={
            'ID': name,
            'GRADE': point
            }
            )
    return { 
        'statusCode': 200,
        'body': json.dumps('Hello ' + name + ', you got '+ str(point)+'!')
    }