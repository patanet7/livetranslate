{
  "basePath": "",
  "documentationLink": "https://developers.google.com/workspace/meet/api",
  "icons": {
    "x32": "http://www.google.com/images/icons/product/search-32.gif",
    "x16": "http://www.google.com/images/icons/product/search-16.gif"
  },
  "parameters": {
    "callback": {
      "description": "JSONP",
      "type": "string",
      "location": "query"
    },
    "fields": {
      "type": "string",
      "location": "query",
      "description": "Selector specifying which fields to include in a partial response."
    },
    "quotaUser": {
      "description": "Available to use for quota purposes for server-side applications. Can be any arbitrary string assigned to a user, but should not exceed 40 characters.",
      "location": "query",
      "type": "string"
    },
    "key": {
      "location": "query",
      "type": "string",
      "description": "API key. Your API key identifies your project and provides you with API access, quota, and reports. Required unless you provide an OAuth 2.0 token."
    },
    "access_token": {
      "description": "OAuth access token.",
      "location": "query",
      "type": "string"
    },
    "oauth_token": {
      "description": "OAuth 2.0 token for the current user.",
      "location": "query",
      "type": "string"
    },
    "prettyPrint": {
      "description": "Returns response with indentations and line breaks.",
      "location": "query",
      "type": "boolean",
      "default": "true"
    },
    "uploadType": {
      "description": "Legacy upload protocol for media (e.g. \"media\", \"multipart\").",
      "location": "query",
      "type": "string"
    },
    "alt": {
      "description": "Data format for response.",
      "enumDescriptions": [
        "Responses with Content-Type of application/json",
        "Media download with context-dependent Content-Type",
        "Responses with Content-Type of application/x-protobuf"
      ],
      "default": "json",
      "enum": [
        "json",
        "media",
        "proto"
      ],
      "type": "string",
      "location": "query"
    },
    "$.xgafv": {
      "description": "V1 error format.",
      "enum": [
        "1",
        "2"
      ],
      "enumDescriptions": [
        "v1 error format",
        "v2 error format"
      ],
      "type": "string",
      "location": "query"
    },
    "upload_protocol": {
      "location": "query",
      "type": "string",
      "description": "Upload protocol for media (e.g. \"raw\", \"multipart\")."
    }
  },
  "ownerDomain": "google.com",
  "protocol": "rest",
  "auth": {
    "oauth2": {
      "scopes": {
        "https://www.googleapis.com/auth/meetings.space.created": {
          "description": "Create, edit, and see information about your Google Meet conferences created by the app."
        },
        "https://www.googleapis.com/auth/meetings.space.readonly": {
          "description": "Read information about any of your Google Meet conferences"
        },
        "https://www.googleapis.com/auth/meetings.space.settings": {
          "description": "Edit, and see settings for all of your Google Meet calls."
        }
      }
    }
  },
  "description": "Create and manage meetings in Google Meet.",
  "version": "v2",
  "name": "meet",
  "id": "meet:v2",
  "mtlsRootUrl": "https://meet.mtls.googleapis.com/",
  "ownerName": "Google",
  "version_module": true,
  "kind": "discovery#restDescription",
  "resources": {
    "spaces": {
      "methods": {
        "get": {
          "flatPath": "v2/spaces/{spacesId}",
          "httpMethod": "GET",
          "scopes": [
            "https://www.googleapis.com/auth/meetings.space.created",
            "https://www.googleapis.com/auth/meetings.space.readonly",
            "https://www.googleapis.com/auth/meetings.space.settings"
          ],
          "description": "Gets details about a meeting space. For an example, see [Get a meeting space](https://developers.google.com/workspace/meet/api/guides/meeting-spaces#get-meeting-space).",
          "path": "v2/{+name}",
          "parameterOrder": [
            "name"
          ],
          "id": "meet.spaces.get",
          "response": {
            "$ref": "Space"
          },
          "parameters": {
            "name": {
              "required": true,
              "type": "string",
              "location": "path",
              "pattern": "^spaces/[^/]+$",
              "description": "Required. Resource name of the space. Format: `spaces/{space}` or `spaces/{meetingCode}`. `{space}` is the resource identifier for the space. It's a unique, server-generated ID and is case sensitive. For example, `jQCFfuBOdN5z`. `{meetingCode}` is an alias for the space. It's a typeable, unique character string and is non-case sensitive. For example, `abc-mnop-xyz`. The maximum length is 128 characters. A `meetingCode` shouldn't be stored long term as it can become dissociated from a meeting space and can be reused for different meeting spaces in the future. Generally, a `meetingCode` expires 365 days after last use. For more information, see [Learn about meeting codes in Google Meet](https://support.google.com/meet/answer/10710509). For more information, see [How Meet identifies a meeting space](https://developers.google.com/workspace/meet/api/guides/meeting-spaces#identify-meeting-space)."
            }
          }
        },
        "endActiveConference": {
          "flatPath": "v2/spaces/{spacesId}:endActiveConference",
          "scopes": [
            "https://www.googleapis.com/auth/meetings.space.created"
          ],
          "request": {
            "$ref": "EndActiveConferenceRequest"
          },
          "id": "meet.spaces.endActiveConference",
          "parameterOrder": [
            "name"
          ],
          "httpMethod": "POST",
          "parameters": {
            "name": {
              "description": "Required. Resource name of the space. Format: `spaces/{space}`. `{space}` is the resource identifier for the space. It's a unique, server-generated ID and is case sensitive. For example, `jQCFfuBOdN5z`. For more information, see [How Meet identifies a meeting space](https://developers.google.com/workspace/meet/api/guides/meeting-spaces#identify-meeting-space).",
              "type": "string",
              "location": "path",
              "pattern": "^spaces/[^/]+$",
              "required": true
            }
          },
          "description": "Ends an active conference (if there's one). For an example, see [End active conference](https://developers.google.com/workspace/meet/api/guides/meeting-spaces#end-active-conference).",
          "path": "v2/{+name}:endActiveConference",
          "response": {
            "$ref": "Empty"
          }
        },
        "patch": {
          "scopes": [
            "https://www.googleapis.com/auth/meetings.space.created",
            "https://www.googleapis.com/auth/meetings.space.settings"
          ],
          "parameterOrder": [
            "name"
          ],
          "response": {
            "$ref": "Space"
          },
          "httpMethod": "PATCH",
          "flatPath": "v2/spaces/{spacesId}",
          "request": {
            "$ref": "Space"
          },
          "parameters": {
            "name": {
              "required": true,
              "pattern": "^spaces/[^/]+$",
              "location": "path",
              "type": "string",
              "description": "Immutable. Resource name of the space. Format: `spaces/{space}`. `{space}` is the resource identifier for the space. It's a unique, server-generated ID and is case sensitive. For example, `jQCFfuBOdN5z`. For more information, see [How Meet identifies a meeting space](https://developers.google.com/workspace/meet/api/guides/meeting-spaces#identify-meeting-space)."
            },
            "updateMask": {
              "description": "Optional. Field mask used to specify the fields to be updated in the space. If update_mask isn't provided(not set, set with empty paths, or only has \"\" as paths), it defaults to update all fields provided with values in the request. Using \"*\" as update_mask will update all fields, including deleting fields not set in the request.",
              "type": "string",
              "location": "query",
              "format": "google-fieldmask"
            }
          },
          "description": "Updates details about a meeting space. For an example, see [Update a meeting space](https://developers.google.com/workspace/meet/api/guides/meeting-spaces#update-meeting-space).",
          "id": "meet.spaces.patch",
          "path": "v2/{+name}"
        },
        "create": {
          "description": "Creates a space.",
          "httpMethod": "POST",
          "parameterOrder": [],
          "path": "v2/spaces",
          "scopes": [
            "https://www.googleapis.com/auth/meetings.space.created"
          ],
          "response": {
            "$ref": "Space"
          },
          "flatPath": "v2/spaces",
          "id": "meet.spaces.create",
          "request": {
            "$ref": "Space"
          },
          "parameters": {}
        }
      }
    },
    "conferenceRecords": {
      "resources": {
        "participants": {
          "methods": {
            "list": {
              "response": {
                "$ref": "ListParticipantsResponse"
              },
              "parameterOrder": [
                "parent"
              ],
              "description": "Lists the participants in a conference record. By default, ordered by join time and in descending order. This API supports `fields` as standard parameters like every other API. However, when the `fields` request parameter is omitted, this API defaults to `'participants/*, next_page_token'`.",
              "flatPath": "v2/conferenceRecords/{conferenceRecordsId}/participants",
              "parameters": {
                "pageToken": {
                  "description": "Page token returned from previous List Call.",
                  "type": "string",
                  "location": "query"
                },
                "filter": {
                  "location": "query",
                  "description": "Optional. User specified filtering condition in [EBNF format](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form). The following are the filterable fields: * `earliest_start_time` * `latest_end_time` For example, `latest_end_time IS NULL` returns active participants in the conference.",
                  "type": "string"
                },
                "parent": {
                  "description": "Required. Format: `conferenceRecords/{conference_record}`",
                  "location": "path",
                  "type": "string",
                  "required": true,
                  "pattern": "^conferenceRecords/[^/]+$"
                },
                "pageSize": {
                  "format": "int32",
                  "type": "integer",
                  "description": "Maximum number of participants to return. The service might return fewer than this value. If unspecified, at most 100 participants are returned. The maximum value is 250; values above 250 are coerced to 250. Maximum might change in the future.",
                  "location": "query"
                }
              },
              "httpMethod": "GET",
              "scopes": [
                "https://www.googleapis.com/auth/meetings.space.created",
                "https://www.googleapis.com/auth/meetings.space.readonly"
              ],
              "path": "v2/{+parent}/participants",
              "id": "meet.conferenceRecords.participants.list"
            },
            "get": {
              "flatPath": "v2/conferenceRecords/{conferenceRecordsId}/participants/{participantsId}",
              "parameterOrder": [
                "name"
              ],
              "response": {
                "$ref": "Participant"
              },
              "httpMethod": "GET",
              "id": "meet.conferenceRecords.participants.get",
              "parameters": {
                "name": {
                  "pattern": "^conferenceRecords/[^/]+/participants/[^/]+$",
                  "type": "string",
                  "required": true,
                  "location": "path",
                  "description": "Required. Resource name of the participant."
                }
              },
              "path": "v2/{+name}",
              "scopes": [
                "https://www.googleapis.com/auth/meetings.space.created",
                "https://www.googleapis.com/auth/meetings.space.readonly"
              ],
              "description": "Gets a participant by participant ID."
            }
          },
          "resources": {
            "participantSessions": {
              "methods": {
                "list": {
                  "parameterOrder": [
                    "parent"
                  ],
                  "path": "v2/{+parent}/participantSessions",
                  "httpMethod": "GET",
                  "response": {
                    "$ref": "ListParticipantSessionsResponse"
                  },
                  "parameters": {
                    "pageToken": {
                      "location": "query",
                      "type": "string",
                      "description": "Optional. Page token returned from previous List Call."
                    },
                    "parent": {
                      "location": "path",
                      "type": "string",
                      "pattern": "^conferenceRecords/[^/]+/participants/[^/]+$",
                      "required": true,
                      "description": "Required. Format: `conferenceRecords/{conference_record}/participants/{participant}`"
                    },
                    "filter": {
                      "type": "string",
                      "location": "query",
                      "description": "Optional. User specified filtering condition in [EBNF format](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form). The following are the filterable fields: * `start_time` * `end_time` For example, `end_time IS NULL` returns active participant sessions in the conference record."
                    },
                    "pageSize": {
                      "format": "int32",
                      "description": "Optional. Maximum number of participant sessions to return. The service might return fewer than this value. If unspecified, at most 100 participants are returned. The maximum value is 250; values above 250 are coerced to 250. Maximum might change in the future.",
                      "type": "integer",
                      "location": "query"
                    }
                  },
                  "description": "Lists the participant sessions of a participant in a conference record. By default, ordered by join time and in descending order. This API supports `fields` as standard parameters like every other API. However, when the `fields` request parameter is omitted this API defaults to `'participantsessions/*, next_page_token'`.",
                  "flatPath": "v2/conferenceRecords/{conferenceRecordsId}/participants/{participantsId}/participantSessions",
                  "id": "meet.conferenceRecords.participants.participantSessions.list",
                  "scopes": [
                    "https://www.googleapis.com/auth/meetings.space.created",
                    "https://www.googleapis.com/auth/meetings.space.readonly"
                  ]
                },
                "get": {
                  "parameters": {
                    "name": {
                      "description": "Required. Resource name of the participant.",
                      "location": "path",
                      "required": true,
                      "type": "string",
                      "pattern": "^conferenceRecords/[^/]+/participants/[^/]+/participantSessions/[^/]+$"
                    }
                  },
                  "flatPath": "v2/conferenceRecords/{conferenceRecordsId}/participants/{participantsId}/participantSessions/{participantSessionsId}",
                  "path": "v2/{+name}",
                  "scopes": [
                    "https://www.googleapis.com/auth/meetings.space.created",
                    "https://www.googleapis.com/auth/meetings.space.readonly"
                  ],
                  "parameterOrder": [
                    "name"
                  ],
                  "response": {
                    "$ref": "ParticipantSession"
                  },
                  "id": "meet.conferenceRecords.participants.participantSessions.get",
                  "description": "Gets a participant session by participant session ID.",
                  "httpMethod": "GET"
                }
              }
            }
          }
        },
        "recordings": {
          "methods": {
            "list": {
              "httpMethod": "GET",
              "parameterOrder": [
                "parent"
              ],
              "flatPath": "v2/conferenceRecords/{conferenceRecordsId}/recordings",
              "response": {
                "$ref": "ListRecordingsResponse"
              },
              "description": "Lists the recording resources from the conference record. By default, ordered by start time and in ascending order.",
              "path": "v2/{+parent}/recordings",
              "id": "meet.conferenceRecords.recordings.list",
              "parameters": {
                "parent": {
                  "type": "string",
                  "pattern": "^conferenceRecords/[^/]+$",
                  "description": "Required. Format: `conferenceRecords/{conference_record}`",
                  "required": true,
                  "location": "path"
                },
                "pageSize": {
                  "location": "query",
                  "description": "Maximum number of recordings to return. The service might return fewer than this value. If unspecified, at most 10 recordings are returned. The maximum value is 100; values above 100 are coerced to 100. Maximum might change in the future.",
                  "type": "integer",
                  "format": "int32"
                },
                "pageToken": {
                  "type": "string",
                  "description": "Page token returned from previous List Call.",
                  "location": "query"
                }
              },
              "scopes": [
                "https://www.googleapis.com/auth/meetings.space.created",
                "https://www.googleapis.com/auth/meetings.space.readonly"
              ]
            },
            "get": {
              "path": "v2/{+name}",
              "id": "meet.conferenceRecords.recordings.get",
              "scopes": [
                "https://www.googleapis.com/auth/meetings.space.created",
                "https://www.googleapis.com/auth/meetings.space.readonly"
              ],
              "description": "Gets a recording by recording ID.",
              "parameterOrder": [
                "name"
              ],
              "httpMethod": "GET",
              "response": {
                "$ref": "Recording"
              },
              "parameters": {
                "name": {
                  "type": "string",
                  "description": "Required. Resource name of the recording.",
                  "location": "path",
                  "required": true,
                  "pattern": "^conferenceRecords/[^/]+/recordings/[^/]+$"
                }
              },
              "flatPath": "v2/conferenceRecords/{conferenceRecordsId}/recordings/{recordingsId}"
            }
          }
        },
        "transcripts": {
          "methods": {
            "list": {
              "id": "meet.conferenceRecords.transcripts.list",
              "response": {
                "$ref": "ListTranscriptsResponse"
              },
              "path": "v2/{+parent}/transcripts",
              "scopes": [
                "https://www.googleapis.com/auth/meetings.space.created",
                "https://www.googleapis.com/auth/meetings.space.readonly"
              ],
              "httpMethod": "GET",
              "flatPath": "v2/conferenceRecords/{conferenceRecordsId}/transcripts",
              "description": "Lists the set of transcripts from the conference record. By default, ordered by start time and in ascending order.",
              "parameters": {
                "pageSize": {
                  "format": "int32",
                  "description": "Maximum number of transcripts to return. The service might return fewer than this value. If unspecified, at most 10 transcripts are returned. The maximum value is 100; values above 100 are coerced to 100. Maximum might change in the future.",
                  "type": "integer",
                  "location": "query"
                },
                "pageToken": {
                  "location": "query",
                  "type": "string",
                  "description": "Page token returned from previous List Call."
                },
                "parent": {
                  "required": true,
                  "type": "string",
                  "pattern": "^conferenceRecords/[^/]+$",
                  "description": "Required. Format: `conferenceRecords/{conference_record}`",
                  "location": "path"
                }
              },
              "parameterOrder": [
                "parent"
              ]
            },
            "get": {
              "flatPath": "v2/conferenceRecords/{conferenceRecordsId}/transcripts/{transcriptsId}",
              "description": "Gets a transcript by transcript ID.",
              "response": {
                "$ref": "Transcript"
              },
              "scopes": [
                "https://www.googleapis.com/auth/meetings.space.created",
                "https://www.googleapis.com/auth/meetings.space.readonly"
              ],
              "httpMethod": "GET",
              "id": "meet.conferenceRecords.transcripts.get",
              "parameters": {
                "name": {
                  "pattern": "^conferenceRecords/[^/]+/transcripts/[^/]+$",
                  "required": true,
                  "type": "string",
                  "description": "Required. Resource name of the transcript.",
                  "location": "path"
                }
              },
              "path": "v2/{+name}",
              "parameterOrder": [
                "name"
              ]
            }
          },
          "resources": {
            "entries": {
              "methods": {
                "get": {
                  "parameterOrder": [
                    "name"
                  ],
                  "scopes": [
                    "https://www.googleapis.com/auth/meetings.space.created",
                    "https://www.googleapis.com/auth/meetings.space.readonly"
                  ],
                  "httpMethod": "GET",
                  "path": "v2/{+name}",
                  "response": {
                    "$ref": "TranscriptEntry"
                  },
                  "description": "Gets a `TranscriptEntry` resource by entry ID. Note: The transcript entries returned by the Google Meet API might not match the transcription found in the Google Docs transcript file. This can occur when 1) we have interleaved speakers within milliseconds, or 2) the Google Docs transcript file is modified after generation.",
                  "parameters": {
                    "name": {
                      "type": "string",
                      "pattern": "^conferenceRecords/[^/]+/transcripts/[^/]+/entries/[^/]+$",
                      "required": true,
                      "location": "path",
                      "description": "Required. Resource name of the `TranscriptEntry`."
                    }
                  },
                  "id": "meet.conferenceRecords.transcripts.entries.get",
                  "flatPath": "v2/conferenceRecords/{conferenceRecordsId}/transcripts/{transcriptsId}/entries/{entriesId}"
                },
                "list": {
                  "parameters": {
                    "parent": {
                      "required": true,
                      "location": "path",
                      "pattern": "^conferenceRecords/[^/]+/transcripts/[^/]+$",
                      "type": "string",
                      "description": "Required. Format: `conferenceRecords/{conference_record}/transcripts/{transcript}`"
                    },
                    "pageToken": {
                      "location": "query",
                      "description": "Page token returned from previous List Call.",
                      "type": "string"
                    },
                    "pageSize": {
                      "description": "Maximum number of entries to return. The service might return fewer than this value. If unspecified, at most 10 entries are returned. The maximum value is 100; values above 100 are coerced to 100. Maximum might change in the future.",
                      "format": "int32",
                      "location": "query",
                      "type": "integer"
                    }
                  },
                  "httpMethod": "GET",
                  "description": "Lists the structured transcript entries per transcript. By default, ordered by start time and in ascending order. Note: The transcript entries returned by the Google Meet API might not match the transcription found in the Google Docs transcript file. This can occur when 1) we have interleaved speakers within milliseconds, or 2) the Google Docs transcript file is modified after generation.",
                  "scopes": [
                    "https://www.googleapis.com/auth/meetings.space.created",
                    "https://www.googleapis.com/auth/meetings.space.readonly"
                  ],
                  "response": {
                    "$ref": "ListTranscriptEntriesResponse"
                  },
                  "id": "meet.conferenceRecords.transcripts.entries.list",
                  "path": "v2/{+parent}/entries",
                  "flatPath": "v2/conferenceRecords/{conferenceRecordsId}/transcripts/{transcriptsId}/entries",
                  "parameterOrder": [
                    "parent"
                  ]
                }
              }
            }
          }
        }
      },
      "methods": {
        "get": {
          "response": {
            "$ref": "ConferenceRecord"
          },
          "path": "v2/{+name}",
          "parameterOrder": [
            "name"
          ],
          "id": "meet.conferenceRecords.get",
          "parameters": {
            "name": {
              "required": true,
              "type": "string",
              "description": "Required. Resource name of the conference.",
              "pattern": "^conferenceRecords/[^/]+$",
              "location": "path"
            }
          },
          "description": "Gets a conference record by conference ID.",
          "scopes": [
            "https://www.googleapis.com/auth/meetings.space.created",
            "https://www.googleapis.com/auth/meetings.space.readonly"
          ],
          "httpMethod": "GET",
          "flatPath": "v2/conferenceRecords/{conferenceRecordsId}"
        },
        "list": {
          "path": "v2/conferenceRecords",
          "parameterOrder": [],
          "response": {
            "$ref": "ListConferenceRecordsResponse"
          },
          "scopes": [
            "https://www.googleapis.com/auth/meetings.space.created",
            "https://www.googleapis.com/auth/meetings.space.readonly"
          ],
          "parameters": {
            "pageSize": {
              "location": "query",
              "format": "int32",
              "description": "Optional. Maximum number of conference records to return. The service might return fewer than this value. If unspecified, at most 25 conference records are returned. The maximum value is 100; values above 100 are coerced to 100. Maximum might change in the future.",
              "type": "integer"
            },
            "pageToken": {
              "type": "string",
              "location": "query",
              "description": "Optional. Page token returned from previous List Call."
            },
            "filter": {
              "description": "Optional. User specified filtering condition in [EBNF format](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form). The following are the filterable fields: * `space.meeting_code` * `space.name` * `start_time` * `end_time` For example, consider the following filters: * `space.name = \"spaces/NAME\"` * `space.meeting_code = \"abc-mnop-xyz\"` * `start_time\u003e=\"2024-01-01T00:00:00.000Z\" AND start_time\u003c=\"2024-01-02T00:00:00.000Z\"` * `end_time IS NULL`",
              "type": "string",
              "location": "query"
            }
          },
          "httpMethod": "GET",
          "description": "Lists the conference records. By default, ordered by start time and in descending order.",
          "id": "meet.conferenceRecords.list",
          "flatPath": "v2/conferenceRecords"
        }
      }
    }
  },
  "rootUrl": "https://meet.googleapis.com/",
  "servicePath": "",
  "schemas": {
    "EndActiveConferenceRequest": {
      "id": "EndActiveConferenceRequest",
      "properties": {},
      "description": "Request to end an ongoing conference of a space.",
      "type": "object"
    },
    "Empty": {
      "description": "A generic empty message that you can re-use to avoid defining duplicated empty messages in your APIs. A typical example is to use it as the request or the response type of an API method. For instance: service Foo { rpc Bar(google.protobuf.Empty) returns (google.protobuf.Empty); }",
      "type": "object",
      "id": "Empty",
      "properties": {}
    },
    "RecordingConfig": {
      "id": "RecordingConfig",
      "description": "Configuration related to recording in a meeting space.",
      "properties": {
        "autoRecordingGeneration": {
          "description": "Defines whether a meeting space is automatically recorded when someone with the privilege to record joins the meeting.",
          "enum": [
            "AUTO_GENERATION_TYPE_UNSPECIFIED",
            "ON",
            "OFF"
          ],
          "type": "string",
          "enumDescriptions": [
            "Default value specified by user policy. This should never be returned.",
            "The artifact is generated automatically.",
            "The artifact is not generated automatically."
          ]
        }
      },
      "type": "object"
    },
    "ListConferenceRecordsResponse": {
      "id": "ListConferenceRecordsResponse",
      "description": "Response of ListConferenceRecords method.",
      "properties": {
        "nextPageToken": {
          "type": "string",
          "description": "Token to be circulated back for further List call if current List does NOT include all the Conferences. Unset if all conferences have been returned."
        },
        "conferenceRecords": {
          "description": "List of conferences in one page.",
          "type": "array",
          "items": {
            "$ref": "ConferenceRecord"
          }
        }
      },
      "type": "object"
    },
    "ListRecordingsResponse": {
      "id": "ListRecordingsResponse",
      "properties": {
        "recordings": {
          "type": "array",
          "items": {
            "$ref": "Recording"
          },
          "description": "List of recordings in one page."
        },
        "nextPageToken": {
          "description": "Token to be circulated back for further List call if current List doesn't include all the recordings. Unset if all recordings are returned.",
          "type": "string"
        }
      },
      "type": "object",
      "description": "Response for ListRecordings method."
    },
    "ListTranscriptEntriesResponse": {
      "description": "Response for ListTranscriptEntries method.",
      "id": "ListTranscriptEntriesResponse",
      "properties": {
        "transcriptEntries": {
          "type": "array",
          "description": "List of TranscriptEntries in one page.",
          "items": {
            "$ref": "TranscriptEntry"
          }
        },
        "nextPageToken": {
          "type": "string",
          "description": "Token to be circulated back for further List call if current List doesn't include all the transcript entries. Unset if all entries are returned."
        }
      },
      "type": "object"
    },
    "ConferenceRecord": {
      "id": "ConferenceRecord",
      "type": "object",
      "description": "Single instance of a meeting held in a space.",
      "properties": {
        "name": {
          "type": "string",
          "description": "Identifier. Resource name of the conference record. Format: `conferenceRecords/{conference_record}` where `{conference_record}` is a unique ID for each instance of a call within a space."
        },
        "expireTime": {
          "readOnly": true,
          "description": "Output only. Server enforced expiration time for when this conference record resource is deleted. The resource is deleted 30 days after the conference ends.",
          "format": "google-datetime",
          "type": "string"
        },
        "startTime": {
          "type": "string",
          "readOnly": true,
          "format": "google-datetime",
          "description": "Output only. Timestamp when the conference started. Always set."
        },
        "endTime": {
          "description": "Output only. Timestamp when the conference ended. Set for past conferences. Unset if the conference is ongoing.",
          "format": "google-datetime",
          "type": "string",
          "readOnly": true
        },
        "space": {
          "type": "string",
          "description": "Output only. The space where the conference was held.",
          "readOnly": true
        }
      }
    },
    "ParticipantSession": {
      "description": "Refers to each unique join or leave session when a user joins a conference from a device. Note that any time a user joins the conference a new unique ID is assigned. That means if a user joins a space multiple times from the same device, they're assigned different IDs, and are also be treated as different participant sessions.",
      "id": "ParticipantSession",
      "type": "object",
      "properties": {
        "name": {
          "description": "Identifier. Session id.",
          "type": "string"
        },
        "endTime": {
          "type": "string",
          "readOnly": true,
          "description": "Output only. Timestamp when the user session ends. Unset if the user session hasn’t ended.",
          "format": "google-datetime"
        },
        "startTime": {
          "type": "string",
          "format": "google-datetime",
          "readOnly": true,
          "description": "Output only. Timestamp when the user session starts."
        }
      }
    },
    "Space": {
      "description": "Virtual place where conferences are held. Only one active conference can be held in one space at any given time.",
      "type": "object",
      "id": "Space",
      "properties": {
        "name": {
          "description": "Immutable. Resource name of the space. Format: `spaces/{space}`. `{space}` is the resource identifier for the space. It's a unique, server-generated ID and is case sensitive. For example, `jQCFfuBOdN5z`. For more information, see [How Meet identifies a meeting space](https://developers.google.com/workspace/meet/api/guides/meeting-spaces#identify-meeting-space).",
          "type": "string"
        },
        "meetingUri": {
          "type": "string",
          "description": "Output only. URI used to join meetings consisting of `https://meet.google.com/` followed by the `meeting_code`. For example, `https://meet.google.com/abc-mnop-xyz`.",
          "readOnly": true
        },
        "meetingCode": {
          "description": "Output only. Type friendly unique string used to join the meeting. Format: `[a-z]+-[a-z]+-[a-z]+`. For example, `abc-mnop-xyz`. The maximum length is 128 characters. Can only be used as an alias of the space name to get the space.",
          "type": "string",
          "readOnly": true
        },
        "config": {
          "description": "Configuration pertaining to the meeting space.",
          "$ref": "SpaceConfig"
        },
        "activeConference": {
          "description": "Active conference, if it exists.",
          "$ref": "ActiveConference"
        }
      }
    },
    "ActiveConference": {
      "type": "object",
      "properties": {
        "conferenceRecord": {
          "type": "string",
          "description": "Output only. Reference to 'ConferenceRecord' resource. Format: `conferenceRecords/{conference_record}` where `{conference_record}` is a unique ID for each instance of a call within a space.",
          "readOnly": true
        }
      },
      "description": "Active conference.",
      "id": "ActiveConference"
    },
    "ListParticipantSessionsResponse": {
      "properties": {
        "nextPageToken": {
          "type": "string",
          "description": "Token to be circulated back for further List call if current List doesn't include all the participants. Unset if all participants are returned."
        },
        "participantSessions": {
          "type": "array",
          "items": {
            "$ref": "ParticipantSession"
          },
          "description": "List of participants in one page."
        }
      },
      "id": "ListParticipantSessionsResponse",
      "description": "Response of ListParticipants method.",
      "type": "object"
    },
    "ListParticipantsResponse": {
      "description": "Response of ListParticipants method.",
      "properties": {
        "totalSize": {
          "description": "Total, exact number of `participants`. By default, this field isn't included in the response. Set the field mask in [SystemParameterContext](https://cloud.google.com/apis/docs/system-parameters) to receive this field in the response.",
          "type": "integer",
          "format": "int32"
        },
        "participants": {
          "description": "List of participants in one page.",
          "items": {
            "$ref": "Participant"
          },
          "type": "array"
        },
        "nextPageToken": {
          "type": "string",
          "description": "Token to be circulated back for further List call if current List doesn't include all the participants. Unset if all participants are returned."
        }
      },
      "type": "object",
      "id": "ListParticipantsResponse"
    },
    "PhoneUser": {
      "type": "object",
      "description": "User dialing in from a phone where the user's identity is unknown because they haven't signed in with a Google Account.",
      "id": "PhoneUser",
      "properties": {
        "displayName": {
          "readOnly": true,
          "description": "Output only. Partially redacted user's phone number when calling.",
          "type": "string"
        }
      }
    },
    "ArtifactConfig": {
      "type": "object",
      "properties": {
        "recordingConfig": {
          "description": "Configuration for recording.",
          "$ref": "RecordingConfig"
        },
        "transcriptionConfig": {
          "$ref": "TranscriptionConfig",
          "description": "Configuration for auto-transcript."
        },
        "smartNotesConfig": {
          "$ref": "SmartNotesConfig",
          "description": "Configuration for auto-smart-notes."
        }
      },
      "id": "ArtifactConfig",
      "description": "Configuration related to meeting artifacts potentially generated by this meeting space."
    },
    "Recording": {
      "id": "Recording",
      "type": "object",
      "properties": {
        "endTime": {
          "description": "Output only. Timestamp when the recording ended.",
          "readOnly": true,
          "format": "google-datetime",
          "type": "string"
        },
        "state": {
          "description": "Output only. Current state.",
          "type": "string",
          "readOnly": true,
          "enum": [
            "STATE_UNSPECIFIED",
            "STARTED",
            "ENDED",
            "FILE_GENERATED"
          ],
          "enumDescriptions": [
            "Default, never used.",
            "An active recording session has started.",
            "This recording session has ended, but the recording file hasn't been generated yet.",
            "Recording file is generated and ready to download."
          ]
        },
        "name": {
          "type": "string",
          "readOnly": true,
          "description": "Output only. Resource name of the recording. Format: `conferenceRecords/{conference_record}/recordings/{recording}` where `{recording}` is a 1:1 mapping to each unique recording session during the conference."
        },
        "driveDestination": {
          "description": "Output only. Recording is saved to Google Drive as an MP4 file. The `drive_destination` includes the Drive `fileId` that can be used to download the file using the `files.get` method of the Drive API.",
          "$ref": "DriveDestination",
          "readOnly": true
        },
        "startTime": {
          "readOnly": true,
          "description": "Output only. Timestamp when the recording started.",
          "type": "string",
          "format": "google-datetime"
        }
      },
      "description": "Metadata about a recording created during a conference."
    },
    "DriveDestination": {
      "id": "DriveDestination",
      "type": "object",
      "properties": {
        "file": {
          "readOnly": true,
          "description": "Output only. The `fileId` for the underlying MP4 file. For example, \"1kuceFZohVoCh6FulBHxwy6I15Ogpc4hP\". Use `$ GET https://www.googleapis.com/drive/v3/files/{$fileId}?alt=media` to download the blob. For more information, see https://developers.google.com/drive/api/v3/reference/files/get.",
          "type": "string"
        },
        "exportUri": {
          "readOnly": true,
          "type": "string",
          "description": "Output only. Link used to play back the recording file in the browser. For example, `https://drive.google.com/file/d/{$fileId}/view`."
        }
      },
      "description": "Export location where a recording file is saved in Google Drive."
    },
    "SmartNotesConfig": {
      "type": "object",
      "properties": {
        "autoSmartNotesGeneration": {
          "description": "Defines whether to automatically generate a summary and recap of the meeting for all invitees in the organization when someone with the privilege to enable smart notes joins the meeting.",
          "enum": [
            "AUTO_GENERATION_TYPE_UNSPECIFIED",
            "ON",
            "OFF"
          ],
          "enumDescriptions": [
            "Default value specified by user policy. This should never be returned.",
            "The artifact is generated automatically.",
            "The artifact is not generated automatically."
          ],
          "type": "string"
        }
      },
      "id": "SmartNotesConfig",
      "description": "Configuration related to smart notes in a meeting space. For more information about smart notes, see [\"Take notes for me\" in Google Meet](https://support.google.com/meet/answer/14754931)."
    },
    "Transcript": {
      "id": "Transcript",
      "properties": {
        "endTime": {
          "type": "string",
          "format": "google-datetime",
          "description": "Output only. Timestamp when the transcript stopped.",
          "readOnly": true
        },
        "startTime": {
          "description": "Output only. Timestamp when the transcript started.",
          "format": "google-datetime",
          "readOnly": true,
          "type": "string"
        },
        "name": {
          "type": "string",
          "readOnly": true,
          "description": "Output only. Resource name of the transcript. Format: `conferenceRecords/{conference_record}/transcripts/{transcript}`, where `{transcript}` is a 1:1 mapping to each unique transcription session of the conference."
        },
        "state": {
          "enum": [
            "STATE_UNSPECIFIED",
            "STARTED",
            "ENDED",
            "FILE_GENERATED"
          ],
          "enumDescriptions": [
            "Default, never used.",
            "An active transcript session has started.",
            "This transcript session has ended, but the transcript file hasn't been generated yet.",
            "Transcript file is generated and ready to download."
          ],
          "description": "Output only. Current state.",
          "type": "string",
          "readOnly": true
        },
        "docsDestination": {
          "$ref": "DocsDestination",
          "description": "Output only. Where the Google Docs transcript is saved.",
          "readOnly": true
        }
      },
      "type": "object",
      "description": "Metadata for a transcript generated from a conference. It refers to the ASR (Automatic Speech Recognition) result of user's speech during the conference."
    },
    "ListTranscriptsResponse": {
      "description": "Response for ListTranscripts method.",
      "id": "ListTranscriptsResponse",
      "properties": {
        "nextPageToken": {
          "description": "Token to be circulated back for further List call if current List doesn't include all the transcripts. Unset if all transcripts are returned.",
          "type": "string"
        },
        "transcripts": {
          "description": "List of transcripts in one page.",
          "items": {
            "$ref": "Transcript"
          },
          "type": "array"
        }
      },
      "type": "object"
    },
    "SpaceConfig": {
      "id": "SpaceConfig",
      "description": "The configuration pertaining to a meeting space.",
      "properties": {
        "entryPointAccess": {
          "description": "Defines the entry points that can be used to join meetings hosted in this meeting space. Default: EntryPointAccess.ALL",
          "enumDescriptions": [
            "Unused.",
            "All entry points are allowed.",
            "Only entry points owned by the Google Cloud project that created the space can be used to join meetings in this space. Apps can use the Meet Embed SDK Web or mobile Meet SDKs to create owned entry points."
          ],
          "type": "string",
          "enum": [
            "ENTRY_POINT_ACCESS_UNSPECIFIED",
            "ALL",
            "CREATOR_APP_ONLY"
          ]
        },
        "attendanceReportGenerationType": {
          "type": "string",
          "enum": [
            "ATTENDANCE_REPORT_GENERATION_TYPE_UNSPECIFIED",
            "GENERATE_REPORT",
            "DO_NOT_GENERATE"
          ],
          "description": "Whether attendance report is enabled for the meeting space.",
          "enumDescriptions": [
            "Default value specified by user policy. This should never be returned.",
            "Attendance report will be generated and sent to drive/email.",
            "Attendance report will not be generated."
          ]
        },
        "moderation": {
          "enum": [
            "MODERATION_UNSPECIFIED",
            "OFF",
            "ON"
          ],
          "type": "string",
          "description": "The pre-configured moderation mode for the Meeting. Default: Controlled by the user's policies.",
          "enumDescriptions": [
            "Moderation type is not specified. This is used to indicate the user hasn't specified any value as the user does not intend to update the state. Users are not allowed to set the value as unspecified.",
            "Moderation is off.",
            "Moderation is on."
          ]
        },
        "accessType": {
          "enumDescriptions": [
            "Default value specified by the user's organization. Note: This is never returned, as the configured access type is returned instead.",
            "Anyone with the join information (for example, the URL or phone access information) can join without knocking.",
            "Members of the host's organization, invited external users, and dial-in users can join without knocking. Everyone else must knock.",
            "Only invitees can join without knocking. Everyone else must knock."
          ],
          "type": "string",
          "enum": [
            "ACCESS_TYPE_UNSPECIFIED",
            "OPEN",
            "TRUSTED",
            "RESTRICTED"
          ],
          "description": "Access type of the meeting space that determines who can join without knocking. Default: The user's default access settings. Controlled by the user's admin for enterprise users or RESTRICTED."
        },
        "moderationRestrictions": {
          "description": "When moderation.ON, these restrictions go into effect for the meeting. When moderation.OFF, will be reset to default ModerationRestrictions.",
          "$ref": "ModerationRestrictions"
        },
        "artifactConfig": {
          "$ref": "ArtifactConfig",
          "description": "Configuration pertaining to the auto-generated artifacts that the meeting supports."
        }
      },
      "type": "object"
    },
    "Participant": {
      "properties": {
        "name": {
          "readOnly": true,
          "description": "Output only. Resource name of the participant. Format: `conferenceRecords/{conference_record}/participants/{participant}`",
          "type": "string"
        },
        "anonymousUser": {
          "description": "Anonymous user.",
          "$ref": "AnonymousUser"
        },
        "latestEndTime": {
          "type": "string",
          "description": "Output only. Time when the participant left the meeting for the last time. This can be null if it's an active meeting.",
          "format": "google-datetime",
          "readOnly": true
        },
        "earliestStartTime": {
          "description": "Output only. Time when the participant first joined the meeting.",
          "format": "google-datetime",
          "type": "string",
          "readOnly": true
        },
        "phoneUser": {
          "$ref": "PhoneUser",
          "description": "User calling from their phone."
        },
        "signedinUser": {
          "description": "Signed-in user.",
          "$ref": "SignedinUser"
        }
      },
      "description": "User who attended or is attending a conference.",
      "type": "object",
      "id": "Participant"
    },
    "TranscriptEntry": {
      "description": "Single entry for one user’s speech during a transcript session.",
      "id": "TranscriptEntry",
      "type": "object",
      "properties": {
        "participant": {
          "type": "string",
          "description": "Output only. Refers to the participant who speaks.",
          "readOnly": true
        },
        "endTime": {
          "type": "string",
          "readOnly": true,
          "format": "google-datetime",
          "description": "Output only. Timestamp when the transcript entry ended."
        },
        "text": {
          "readOnly": true,
          "description": "Output only. The transcribed text of the participant's voice, at maximum 10K words. Note that the limit is subject to change.",
          "type": "string"
        },
        "name": {
          "description": "Output only. Resource name of the entry. Format: \"conferenceRecords/{conference_record}/transcripts/{transcript}/entries/{entry}\"",
          "type": "string",
          "readOnly": true
        },
        "startTime": {
          "readOnly": true,
          "type": "string",
          "format": "google-datetime",
          "description": "Output only. Timestamp when the transcript entry started."
        },
        "languageCode": {
          "type": "string",
          "readOnly": true,
          "description": "Output only. Language of spoken text, such as \"en-US\". IETF BCP 47 syntax (https://tools.ietf.org/html/bcp47)"
        }
      }
    },
    "DocsDestination": {
      "type": "object",
      "description": "Google Docs location where the transcript file is saved.",
      "id": "DocsDestination",
      "properties": {
        "document": {
          "description": "Output only. The document ID for the underlying Google Docs transcript file. For example, \"1kuceFZohVoCh6FulBHxwy6I15Ogpc4hP\". Use the `documents.get` method of the Google Docs API (https://developers.google.com/docs/api/reference/rest/v1/documents/get) to fetch the content.",
          "type": "string",
          "readOnly": true
        },
        "exportUri": {
          "type": "string",
          "description": "Output only. URI for the Google Docs transcript file. Use `https://docs.google.com/document/d/{$DocumentId}/view` to browse the transcript in the browser.",
          "readOnly": true
        }
      }
    },
    "ModerationRestrictions": {
      "description": "Defines restrictions for features when the meeting is moderated.",
      "type": "object",
      "id": "ModerationRestrictions",
      "properties": {
        "reactionRestriction": {
          "enumDescriptions": [
            "Default value specified by user policy. This should never be returned.",
            "Meeting owner and co-host have the permission.",
            "All Participants have permissions."
          ],
          "type": "string",
          "enum": [
            "RESTRICTION_TYPE_UNSPECIFIED",
            "HOSTS_ONLY",
            "NO_RESTRICTION"
          ],
          "description": "Defines who has permission to send reactions in the meeting space."
        },
        "presentRestriction": {
          "enumDescriptions": [
            "Default value specified by user policy. This should never be returned.",
            "Meeting owner and co-host have the permission.",
            "All Participants have permissions."
          ],
          "type": "string",
          "enum": [
            "RESTRICTION_TYPE_UNSPECIFIED",
            "HOSTS_ONLY",
            "NO_RESTRICTION"
          ],
          "description": "Defines who has permission to share their screen in the meeting space."
        },
        "chatRestriction": {
          "type": "string",
          "enumDescriptions": [
            "Default value specified by user policy. This should never be returned.",
            "Meeting owner and co-host have the permission.",
            "All Participants have permissions."
          ],
          "description": "Defines who has permission to send chat messages in the meeting space.",
          "enum": [
            "RESTRICTION_TYPE_UNSPECIFIED",
            "HOSTS_ONLY",
            "NO_RESTRICTION"
          ]
        },
        "defaultJoinAsViewerType": {
          "description": "Defines whether to restrict the default role assigned to users as viewer.",
          "type": "string",
          "enumDescriptions": [
            "Default value specified by user policy. This should never be returned.",
            "Users will by default join as viewers.",
            "Users will by default join as contributors."
          ],
          "enum": [
            "DEFAULT_JOIN_AS_VIEWER_TYPE_UNSPECIFIED",
            "ON",
            "OFF"
          ]
        }
      }
    },
    "AnonymousUser": {
      "properties": {
        "displayName": {
          "description": "Output only. User provided name when they join a conference anonymously.",
          "readOnly": true,
          "type": "string"
        }
      },
      "description": "User who joins anonymously (meaning not signed into a Google Account).",
      "id": "AnonymousUser",
      "type": "object"
    },
    "SignedinUser": {
      "description": "A signed-in user can be: a) An individual joining from a personal computer, mobile device, or through companion mode. b) A robot account used by conference room devices.",
      "properties": {
        "user": {
          "readOnly": true,
          "description": "Output only. Unique ID for the user. Interoperable with Admin SDK API and People API. Format: `users/{user}`",
          "type": "string"
        },
        "displayName": {
          "description": "Output only. For a personal device, it's the user's first name and last name. For a robot account, it's the administrator-specified device name. For example, \"Altostrat Room\".",
          "readOnly": true,
          "type": "string"
        }
      },
      "type": "object",
      "id": "SignedinUser"
    },
    "TranscriptionConfig": {
      "type": "object",
      "properties": {
        "autoTranscriptionGeneration": {
          "type": "string",
          "enum": [
            "AUTO_GENERATION_TYPE_UNSPECIFIED",
            "ON",
            "OFF"
          ],
          "description": "Defines whether the content of a meeting is automatically transcribed when someone with the privilege to transcribe joins the meeting.",
          "enumDescriptions": [
            "Default value specified by user policy. This should never be returned.",
            "The artifact is generated automatically.",
            "The artifact is not generated automatically."
          ]
        }
      },
      "description": "Configuration related to transcription in a meeting space.",
      "id": "TranscriptionConfig"
    }
  },
  "title": "Google Meet API",
  "canonicalName": "Meet",
  "baseUrl": "https://meet.googleapis.com/",
  "fullyEncodeReservedExpansion": true,
  "batchPath": "batch",
  "revision": "20250625",
  "discoveryVersion": "v1"
}