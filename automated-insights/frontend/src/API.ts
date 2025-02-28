/* tslint:disable */
/* eslint-disable */
//  This file was automatically generated and should not be edited.

export type CreateConversationPayload = {
  __typename: "CreateConversationPayload",
  conversation?: Conversation | null,
};

export type Conversation = {
  __typename: "Conversation",
  conversationId: string,
  userId: string,
  messages?:  Array<Message > | null,
  status: ConversationStatus,
  createdAt: string,
};

export type Message = {
  __typename: "Message",
  sender: string,
  message: string,
  createdAt: string,
};

// ############# ENUMS #####################
export enum ConversationStatus {
  NEW = "NEW",
  PENDING = "PENDING",
  PROCESSING = "PROCESSING",
  COMPLETE = "COMPLETE",
  ERROR = "ERROR",
}


export type CreateMessageInput = {
  conversationId: string,
  prompt: string,
};

export type CreateMessagePayload = {
  __typename: "CreateMessagePayload",
  message?: Message | null,
};

export type DeleteConversationInput = {
  conversationId: string,
};

export type DeleteConversationPayload = {
  __typename: "DeleteConversationPayload",
  conversation?: Conversation | null,
};

export type SystemSendMessageChunkInput = {
  userId: string,
  conversationId: string,
  status: ConversationStatus,
  chunkType: string,
  chunk: string,
};

export type MessageChunk = {
  __typename: "MessageChunk",
  userId: string,
  conversationId: string,
  status: ConversationStatus,
  chunkType: string,
  chunk: string,
};

export type GetConversationInput = {
  conversationId: string,
};

export type RecieveMessageChunkAsyncInput = {
  conversationId: string,
};

export type CreateConversationMutationVariables = {
};

export type CreateConversationMutation = {
  // Creation
  createConversation?:  {
    __typename: "CreateConversationPayload",
    conversation?:  {
      __typename: "Conversation",
      conversationId: string,
      userId: string,
      status: ConversationStatus,
      createdAt: string,
    } | null,
  } | null,
};

export type CreateMessageAsyncMutationVariables = {
  input: CreateMessageInput,
};

export type CreateMessageAsyncMutation = {
  createMessageAsync?:  {
    __typename: "CreateMessagePayload",
    message?:  {
      __typename: "Message",
      sender: string,
      message: string,
      createdAt: string,
    } | null,
  } | null,
};

export type DeleteConversationMutationVariables = {
  input: DeleteConversationInput,
};

export type DeleteConversationMutation = {
  // Deletion
  deleteConversation?:  {
    __typename: "DeleteConversationPayload",
    conversation?:  {
      __typename: "Conversation",
      conversationId: string,
      userId: string,
      status: ConversationStatus,
      createdAt: string,
    } | null,
  } | null,
};

export type SystemSendMessageChunkMutationVariables = {
  input: SystemSendMessageChunkInput,
};

export type SystemSendMessageChunkMutation = {
  // System
  systemSendMessageChunk?:  {
    __typename: "MessageChunk",
    userId: string,
    conversationId: string,
    status: ConversationStatus,
    chunkType: string,
    chunk: string,
  } | null,
};

export type GetConversationQueryVariables = {
  input: GetConversationInput,
};

export type GetConversationQuery = {
  getConversation?:  {
    __typename: "Conversation",
    conversationId: string,
    userId: string,
    messages?:  Array< {
      __typename: "Message",
      sender: string,
      message: string,
      createdAt: string,
    } > | null,
    status: ConversationStatus,
    createdAt: string,
  } | null,
};

export type GetAllConversationsQueryVariables = {
};

export type GetAllConversationsQuery = {
  getAllConversations?:  Array< {
    __typename: "Conversation",
    conversationId: string,
    userId: string,
    messages?:  Array< {
      __typename: "Message",
      sender: string,
      message: string,
      createdAt: string,
    } > | null,
    status: ConversationStatus,
    createdAt: string,
  } > | null,
};

export type RecieveMessageChunkAsyncSubscriptionVariables = {
  input: RecieveMessageChunkAsyncInput,
};

export type RecieveMessageChunkAsyncSubscription = {
  recieveMessageChunkAsync?:  {
    __typename: "MessageChunk",
    userId: string,
    conversationId: string,
    status: ConversationStatus,
    chunkType: string,
    chunk: string,
  } | null,
};
