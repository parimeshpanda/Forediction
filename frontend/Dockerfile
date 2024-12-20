FROM node:18
WORKDIR /usr/src/app
COPY package*.json ./
RUN npm config set strict-ssl false 
RUN npm config set registry https://registry.npmjs.org/
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "start"]