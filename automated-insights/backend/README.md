# GenAI Backend

## Deployment

### Install dependencies
```
cd backend
npm install
```

### Login / Set your AWS profile
```
export AWS_PROFILE=team
export AWS_REGION=us-east-1
```

### Set Athena database and Bedrock Agent ID and Bedrock Agent Alias ID of the Supervisor Agent in the `lib/genai-app-stack.ts` file. Update AGENT_ID, AGENT_ALIAS_ID, ATHENA_DATABASE_NAME, and BUCKET_NAME.
```
environment: {
        GRAPHQL_URL: api.graphqlUrl,
        TABLE_NAME: table.tableName,
        TRACE_TABLE_NAME: trace_table.tableName,
        USER_FEEDBACK_TABLE_NAME: userfeedback_table.tableName,
        REGION: awsRegion,
        USER_UPLOAD_BUCKET_NAME: userUploadBucket.bucketName,
        WEBSITE_BUCKET_NAME: websiteBucket.bucketName,
        BUCKET_NAME: 'bucket-name',
        FLOW_ID: 'flow-id',
        FLOW_ALIAS_ID: 'flow-alias-id',
        AGENT_ID: 'agent-id',
        AGENT_ALIAS_ID: 'agent-alias-id',
        ATHENA_DATABASE_NAME: 'athena-database-name',
      },
```

### Bootstrap and deploy
```
cdk bootstrap
npm run deploy
```