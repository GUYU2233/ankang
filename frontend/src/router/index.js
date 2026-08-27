import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', redirect: '/dashboard' },
  { path: '/dashboard', name: 'dashboard', component: () => import('../views/Dashboard.vue'), meta: { title: '数据看板' } },
  { path: '/devices', name: 'devices', component: () => import('../views/Devices.vue'), meta: { title: '设备接入' } },
  { path: '/residents', name: 'residents', component: () => import('../views/Residents.vue'), meta: { title: '老人档案' } },
  { path: '/alerts', name: 'alerts', component: () => import('../views/Alerts.vue'), meta: { title: '告警中心' } },
  { path: '/live', name: 'live', component: () => import('../views/Live.vue'), meta: { title: '实时画面' } }
]

export default createRouter({ history: createWebHistory(), routes })