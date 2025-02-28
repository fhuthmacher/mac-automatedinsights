/* tslint:disable */
/* eslint-disable */
// this is an auto generated file. This will be overwritten

import * as APITypes from "../API";
type GeneratedQuery<InputType, OutputType> = string & {
  __generatedQueryInput: InputType;
  __generatedQueryOutput: OutputType;
};

export const getConversation = /* GraphQL */ `query GetConversation($input: GetConversationInput!) {
  getConversation(input: $input) {
    conversationId
    userId
    messages {
      sender
      message
      createdAt
      citations {
        content
        imageContent
        imageUrl
        location {
          type
          uri
          url
        }
        metadata
        span
      }
    }
    status
    createdAt
  }
}
` as GeneratedQuery<APITypes.GetConversationQueryVariables, APITypes.GetConversationQuery>;
export const getAllConversations = /* GraphQL */ `query GetAllConversations {
  getAllConversations {
    conversationId
    userId
    messages {
      sender
      message
      createdAt
      citations {
        content
        imageContent
        imageUrl
        location {
          type
          uri
          url
        }
        metadata
        span
      }
    }
    status
    createdAt
  }
}
` as GeneratedQuery<
  APITypes.GetAllConversationsQueryVariables,
  APITypes.GetAllConversationsQuery
>;
