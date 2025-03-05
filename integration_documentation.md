# Bot Framework Integration Guide

## Introduction

This guide explains how to integrate your adapter service with the Bot Framework. The Bot Framework provides a modular, extensible architecture for building conversational AI agents that can connect to various messaging platforms through normalizing adapters.

## What is a Normalizing Adapter?

A normalizing adapter is a service that translates platform-specific message formats (from Slack, Discord, etc.) into a standardized format that the Bot Framework can process. It acts as a bridge between external messaging platforms and the Bot Framework.

## Integration Overview

Integration with the Bot Framework happens through Socket.IO connections. Your adapter will:

1. Accept connections from the Bot Framework
2. Send standardized events to the Bot Framework
3. Receive responses and commands from the Bot Framework
4. Translate these responses back to your platform's format

## Socket.IO Event Protocol

### Events from Adapter to Bot Framework

#### `chat_message`

Send this event when a user sends a message that should be processed by the bot.

```javascript
{
  "event_type": "chat_message",  // Type of event
  "chat_id": "C123456",          // Conversation identifier
  "user_id": "U123456",          // User identifier
  "content": "Hello bot!",       // Message content
  "message_id": "M123456",       // Optional: Message identifier
  "platform": "slack",           // Platform identifier
  "sender": {                    // Optional: Sender information
    "userId": "U123456",
    "displayName": "John Doe"
  },
  "threadId": "T123456",         // Optional: Thread identifier
  "attachments": [],             // Optional: Array of attachments
  "timestamp": 1675703895000     // Optional: Message timestamp
}
```

#### `clear_context`

Send this event when you want to clear the conversation context for a specific environment.

```javascript
{
  "env_id": "messaging_C123456",  // Environment identifier
  "adapter_id": "slack_adapter"   // Your adapter identifier
}
```

### Events from Bot Framework to Adapter

#### `bot_response`

The Bot Framework sends this event when it has a response to a user message.

```javascript
{
  "adapter_id": "slack_adapter",  // Your adapter identifier
  "event": "bot_response",        // Event type
  "data": {
    "user_id": "U123456",         // User identifier
    "message": "Hello user!",     // Response content
    "message_id": "M123456"       // Optional: Message identifier for threading
  }
}
```

#### `typing_indicator`

The Bot Framework sends this event to indicate the bot is typing.

```javascript
{
  "adapter_id": "slack_adapter",  // Your adapter identifier
  "event": "typing_indicator",    // Event type
  "data": {
    "chat_id": "C123456",         // Conversation identifier
    "is_typing": true             // Whether the bot is typing
  }
}
```

#### `error`

The Bot Framework sends this event when an error occurs.

```javascript
{
  "adapter_id": "slack_adapter",  // Your adapter identifier
  "event": "error",               // Event type
  "data": {
    "chat_id": "C123456",         // Conversation identifier
    "error": "Error message"      // Error description
  }
}
```

## Registration Process

When your adapter starts, it should listen for connections from the Bot Framework. The Bot Framework will connect to your adapter and send a registration event.

### `register_bot`

The Bot Framework sends this event to register with your adapter.

```javascript
{
  "bot_id": "my_bot",             // Bot identifier
  "bot_name": "My Bot",           // Bot name
  "capabilities": ["chat", "file"] // Bot capabilities
}
```

Your adapter should respond with either `registration_success` or `registration_error`.

### `registration_success`

Send this event when the bot has been successfully registered with your adapter.

```javascript
{
  "adapter_id": "slack_adapter",   // Your adapter identifier
  "platform": "slack",             // Platform identifier
  "status": "success",             // Registration status
  "message": "Bot registered successfully"
}
```

### `registration_error`

Send this event when there was an error registering the bot.

```javascript
{
  "adapter_id": "slack_adapter",   // Your adapter identifier
  "platform": "slack",             // Platform identifier
  "status": "error",               // Registration status
  "message": "Error registering bot: Invalid credentials"
}
```

## Standardized Message Format

The Bot Framework expects messages in a standardized format. Your adapter should translate platform-specific formats to this standard:

### Incoming Messages (from platform to Bot Framework)

```javascript
{
  "event": "messageReceived",
  "data": {
    "messageId": "M123456",
    "conversationId": "C123456",
    "sender": {
      "userId": "U123456",
      "displayName": "John Doe"
    },
    "text": "Hello bot!",
    "threadId": "T123456",
    "attachments": [],
    "timestamp": 1675703895000,
    "adapter_id": "slack_adapter"
  }
}
```

### Outgoing Messages (from Bot Framework to platform)

```javascript
{
  "action": "sendMessage",
  "payload": {
    "conversationId": "C123456",
    "text": "Hello user!",
    "threadId": "T123456"
  },
  "adapter_id": "slack_adapter"
}
```

## Implementation Example

Here's a simple example of how to implement a Socket.IO server for your adapter:

```javascript
const { Server } = require('socket.io');
const io = new Server(3000);

// Store connected bots
const connectedBots = {};

io.on('connection', (socket) => {
  console.log('Bot Framework connected');
  
  // Handle bot registration
  socket.on('register_bot', (data) => {
    const botId = data.bot_id;
    connectedBots[botId] = {
      socket: socket,
      name: data.bot_name,
      capabilities: data.capabilities
    };
    
    // Send registration success
    socket.emit('registration_success', {
      adapter_id: 'my_adapter',
      platform: 'my_platform',
      status: 'success',
      message: 'Bot registered successfully'
    });
  });
  
  // Handle bot responses
  socket.on('bot_response', (data) => {
    const userId = data.data.user_id;
    const message = data.data.message;
    
    // Translate and send to your platform
    console.log(`Sending to user ${userId}: ${message}`);
    // yourPlatformAPI.sendMessage(userId, message);
  });
  
  // Handle typing indicators
  socket.on('typing_indicator', (data) => {
    const chatId = data.data.chat_id;
    const isTyping = data.data.is_typing;
    
    // Send typing indicator to your platform
    console.log(`Bot is ${isTyping ? 'typing' : 'not typing'} in chat ${chatId}`);
    // yourPlatformAPI.sendTypingIndicator(chatId, isTyping);
  });
  
  // Handle disconnection
  socket.on('disconnect', () => {
    // Remove bot from connected bots
    for (const [botId, bot] of Object.entries(connectedBots)) {
      if (bot.socket === socket) {
        delete connectedBots[botId];
        console.log(`Bot ${botId} disconnected`);
      }
    }
  });
});

// Example: When a message is received from your platform
function onPlatformMessage(userId, chatId, text, messageId) {
  // Find a bot to handle this message
  const botId = Object.keys(connectedBots)[0]; // Simple example - use your own logic
  if (!botId) return;
  
  const bot = connectedBots[botId];
  
  // Send standardized message to Bot Framework
  bot.socket.emit('chat_message', {
    event_type: 'chat_message',
    chat_id: chatId,
    user_id: userId,
    content: text,
    message_id: messageId,
    platform: 'my_platform'
  });
}
```
