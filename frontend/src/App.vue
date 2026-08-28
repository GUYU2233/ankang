<template>
  <el-container class="app-root">
    <el-aside width="200px" class="sidebar">
      <div class="brand">智护安康</div>
      <el-menu :default-active="$route.path" router class="nav">
        <el-menu-item index="/dashboard">数据看板</el-menu-item>
        <el-menu-item index="/devices">设备接入</el-menu-item>
        <el-menu-item index="/residents">老人档案</el-menu-item>
        <el-menu-item index="/live">实时画面</el-menu-item>
        <el-menu-item index="/multimodal">视觉巡检</el-menu-item>
        <el-menu-item index="/alerts">告警中心</el-menu-item>
      </el-menu>
    </el-aside>
    <el-container>
      <el-header class="topbar">
        <div class="page-title">{{ $route.meta.title || '' }}</div>
        <el-tag v-if="health" size="small" type="success" effect="plain">服务正常</el-tag>
        <el-tag v-else size="small" type="danger" effect="plain">服务离线</el-tag>
      </el-header>
      <el-main class="main-body">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue'
import api from './api'

const health = ref(false)
let timer = null

async function checkHealth() {
  try { await api.health(); health.value = true } catch (e) { health.value = false }
}

onMounted(() => { checkHealth(); timer = setInterval(checkHealth, 10000) })
onBeforeUnmount(() => { if (timer) clearInterval(timer) })
</script>

<style>
html, body, #app { height: 100%; margin: 0; padding: 0; background: #f5f6f8; color: #303133; font-size: 14px; }
.app-root { height: 100%; }
.sidebar { background: #fff; border-right: 1px solid #e5e6eb; }
.brand { padding: 16px; font-size: 16px; font-weight: 600; border-bottom: 1px solid #f0f1f3; }
.nav { border-right: none; }
.topbar { display: flex; align-items: center; justify-content: space-between; background: #fff; border-bottom: 1px solid #e5e6eb; }
.page-title { font-size: 16px; font-weight: 600; }
.main-body { padding: 16px; }
</style>
