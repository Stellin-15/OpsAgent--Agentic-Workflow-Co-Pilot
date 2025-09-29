# SOP-001: Standard Procedure for Restarting a Web Server

## 1. Identify the Server
First, confirm the hostname or IP address of the server that needs to be restarted. Example: `web-prod-03`.

## 2. Announce Maintenance
Post a message in the #ops-alerts Slack channel. 
Message template: "NOTICE: Beginning a scheduled restart of server `[SERVER_HOSTNAME]`. A brief service interruption is expected."

## 3. SSH into the Server
Use your standard credentials to connect to the server.
`ssh admin@[SERVER_HOSTNAME]`

## 4. Execute the Restart Command
Use the system's service manager to perform a graceful restart.
`sudo systemctl restart nginx`

## 5. Verify Service Restoration
After 30-60 seconds, check that the service is back online. You can use `curl` or check the service status.
`curl http://localhost`
`sudo systemctl status nginx`
A successful check should return an HTTP 200 OK or show an "active (running)" status.

## 6. Announce Completion
Post a follow-up message in the #ops-alerts Slack channel.
Message template: "COMPLETED: Restart of server `[SERVER_HOSTNAME]` is complete. Service is restored."