**Installation**

!pip install -r requirements.txt
!python install-model.py

**MYSQL SETUP:**
INSTALLATION:
sudo apt install mysql-server

sudo mysql_secure_installation

to get started with MySQL go to the root directory.  
sudo mysql -u root

-for creating database name-
create database database_name;


-reset password for mysql-
SELECT user, host, authentication_string, plugin FROM mysql.user WHERE user = 'root';
ALTER USER 'root'@'localhost' IDENTIFIED WITH 'mysql_native_password' BY 'new_password';
SHOW GRANTS FOR 'root'@'localhost';
sudo service mysql restart
ALTER USER 'root'@'localhost' IDENTIFIED BY 'new_password';
or 
ALTER USER 'root'@'localhost' IDENTIFIED BY 'password';


-for starting mysql-
sudo service mysql start

-for status-
sudo service mysql status

-for activating 3306-
sudo ufw allow 3306



**NGROK SETUP:**

Add-Authtoken
```
ngrok config add-authtoken 2XC1R2cPtrhJ53YYr7ksgt3Mlom_6UMa9moqQT7nN3ZRBZL1a
```
Ngrok multiuser
```
version: 2
authtoken: 2XC1R2cPtrhJ53YYr7ksgt3Mlom_6UMa9moqQT7nN3ZRBZL1a
tunnels:
  first:
    proto: http
    addr: https://localhost:8080
  second:
    proto: http
    addr: 8000
```
