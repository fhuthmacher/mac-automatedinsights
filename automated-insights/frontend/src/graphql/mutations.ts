/* tslint:disable */
/* eslint-disable */
// this is an auto generated file. This will be overwritten

import * as APITypes from "../API";
type GeneratedMutation<InputType, OutputType> = string & {
  __generatedMutationInput: InputType;
  __generatedMutationOutput: OutputType;
};

export const createConversation = /* GraphQL */ `mutation CreateConversation {
  createConversation {
    conversation {
      conversationId
      userId
      status
      createdAt
      __typename
    }
    __typename
  }
}
` as GeneratedMutation<
  APITypes.CreateConversationMutationVariables,
  APITypes.CreateConversationMutation
>;
export const createMessageAsync = /* GraphQL */ `mutation CreateMessageAsync($input: CreateMessageInput!) {
  createMessageAsync(input: $input) {
    message {
      sender
      message
      createdAt
      __typename
    }
    __typename
  }
}
` as GeneratedMutation<
  APITypes.CreateMessageAsyncMutationVariables,
  APITypes.CreateMessageAsyncMutation
>;
export const deleteConversation = /* GraphQL */ `mutation DeleteConversation($input: DeleteConversationInput!) {
  deleteConversation(input: $input) {
    conversation {
      conversationId
      userId
      status
      createdAt
      __typename
    }
    __typename
  }
}
` as GeneratedMutation<
  APITypes.DeleteConversationMutationVariables,
  APITypes.DeleteConversationMutation
>;
export const systemSendMessageChunk = /* GraphQL */ `mutation SystemSendMessageChunk($input: SystemSendMessageChunkInput!) {
  systemSendMessageChunk(input: $input) {
    userId
    conversationId
    status
    chunkType
    chunk
    __typename
  }
}
` as GeneratedMutation<
  APITypes.SystemSendMessageChunkMutationVariables,
  APITypes.SystemSendMessageChunkMutation
>;
